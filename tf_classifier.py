import os, sys
sys.path.append(os.path.join(".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DatasetFromDirectory, get_normalization_parameters

from train import get_required_transforms
from config import PATCH_SIZE, NUMBER_OF_BANDS, TEST_DATASETS, TEST_ROOT_DATASET_DIR, DATASET_NAME, batch_size, end_epoch

num_classes = len(TEST_DATASETS)
input_shape = (PATCH_SIZE, PATCH_SIZE, NUMBER_OF_BANDS)

input_transform, label_transform = get_required_transforms(task="classification")
dataset = DatasetFromDirectory(root=TEST_ROOT_DATASET_DIR,
							   dataset_name=DATASET_NAME,
							   task="classification",
							   patch_size=PATCH_SIZE,
							   lazy_read=False,
							   shuffle=True,
							   rgbn_from_cube=False,
							   product_pairing=False,
							   train_with_patches=True,
							   positive_only=True,
							   verbose=True,
							   transform=(input_transform, label_transform))

data = DataLoader(dataset=dataset, shuffle=True)

X, y = [],[]

for image, hypercube, classlabel in data:
	hypercube = hypercube.numpy()
	hypercube = np.transpose(hypercube, [0, 2, 3, 1]).squeeze(0)

	X.append(hypercube)
	y.append(classlabel.ravel())

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

""" Configure the hyperparameters """
weight_decay = 1e-6
batch_size = batch_size
num_epochs = end_epoch
dropout_rate = 0.15
image_size = 64								# We'll resize input images to this size.
patch_size = 16								# Size of the patches to be extracted from the input images.
num_patches = (image_size//patch_size) ** 2	# Size of the data array.
embedding_dim = 256							# Number of hidden units.
num_blocks = 6								# Number of blocks.

print(f"\nImage Size ({image_size} x {image_size}):\t\t\t{image_size ** 2}")
print(f"Patch Size ({patch_size} x {patch_size}):\t\t\t{patch_size ** 2}")
print(f"Patches per Image ({image_size}/{patch_size} * {image_size}/{patch_size}):\t{num_patches}")
print(f"Elements per Patch ({num_patches}*{num_patches}*{NUMBER_OF_BANDS} Channels):\t{(patch_size ** 2) * NUMBER_OF_BANDS}")

def build_classifier(blocks, positional_encoding=False):
	""" Build a classification model.
	We implement a method that builds a classifier given the processing blocks. """
	inputs = layers.Input(shape=input_shape)
	augmented = data_augmentation(inputs)
	patches = Patches(patch_size, num_patches)(augmented)
	# print("Before Patching and After Patching:", augmented.shape, patches.shape)
	x = layers.Dense(units=embedding_dim)(patches)				# [batch_size, num_patches, embedding_dim] tensor
	if positional_encoding:
		positions = tf.range(start=0, limit=num_patches, delta=1)
		position_embedding = layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)(positions)
		x = x + position_embedding
	x = blocks(x)
	representation = layers.GlobalAveragePooling1D()(x)			# [batch_size, embedding_dim] representation tensor
	# print("Before and After Global Averaging of the Patches:", x.shape, representation.shape)
	representation = layers.Dropout(rate=dropout_rate)(representation)
	logits = layers.Dense(num_classes)(representation)
	return keras.Model(inputs=inputs, outputs=logits)

def run_experiment(model):
	"""	Define an experiment.
	We implement a utility function to compile, train, and evaluate a given model. """
	optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
	model.compile(optimizer=optimizer,
				  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

	print("\n" + 33*"-" + "Training" + 34*"-" + "\n")
	history = model.fit(x=X_train, y=y_train, batch_size=batch_size,
						epochs=num_epochs, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
	for item in y_test:
		print(item, end=" ")

	_, accuracy = model.evaluate(X_test, y_test)
	print(f"Test accuracy: {round(accuracy * 100, 2)}%")

	return history

""" Use data augmentation """
data_augmentation = keras.Sequential([layers.experimental.preprocessing.Resizing(image_size, image_size),
									  layers.experimental.preprocessing.RandomFlip("horizontal"),
									  layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2)],
									 name="data_augmentation")

# Compute the mean and the variance of the training data for normalization.
# data_augmentation.layers[0].adapt(X_train)
# data_augmentation.layers[0].adapt(X_test)

class Patches(layers.Layer):
	""" Implement patch extraction as a layer """
	def __init__(self, patch_size, num_patches):
		super(Patches, self).__init__()
		self.patch_size = patch_size
		self.num_patches = num_patches

	def call(self, images):
		batch_size = tf.shape(images)[0]
		patches = tf.image.extract_patches(images=images,
										   sizes=[1, self.patch_size, self.patch_size, 1],
										   strides=[1, self.patch_size, self.patch_size, 1],
										   rates=[1, 1, 1, 1],
										   padding="VALID")

		patch_dims = patches.shape[-1]
		patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
		return patches

class FNetLayer(layers.Layer):
	""" Implement the FNet module """
	def __init__(self, embedding_dim, dropout_rate, *args, **kwargs):
		super(FNetLayer, self).__init__(*args, **kwargs)

		self.ffn = keras.Sequential([layers.Dense(units=embedding_dim),
									 tfa.layers.GELU(),
									 layers.Dropout(rate=dropout_rate),
									 layers.Dense(units=embedding_dim)])

		self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
		self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		# Apply fourier transformations.
		x = tf.cast(tf.signal.fft2d(tf.cast(inputs, dtype=tf.dtypes.complex64)), dtype=tf.dtypes.float32)
		# Add skip connection.
		x = x + inputs
		# Apply layer normalization.
		x = self.normalize1(x)
		# Apply Feedfowrad network.
		x_ffn = self.ffn(x)
		# Add skip connection.
		x = x + x_ffn
		# Apply layer normalization.
		return self.normalize2(x)

fnet_blocks = keras.Sequential([FNetLayer(embedding_dim, dropout_rate) for _ in range(num_blocks)])

learning_rate = 0.001
fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
history = run_experiment(fnet_classifier)