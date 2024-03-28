# Assessing Fruit Ripeness using Smartphones

## Spectral Upsampling
The Spectral Upsampling model is built on PyTorch and is trained to produce hyperspectral cubes when the input to it is an RGB and NIR image.

In order to reproduce our results using our pre-trained models or the models can be trained from scratch.

### Installation
Clone the repository
...

The directory tree should look something like this (Some directories are created when the scripts are run):

```
MobiSLP
│
└── checkpoints			Pretrained models
|
└── dataPreparation		Scripts to prepare the data
│
└── logs			Training and testing logs
|
└── models			Model Architecture Scripts
|
└── visualizationsScripts	Contains scripts to produce various visualizations
|
└── visualizations		Contains output of said scripts
```

### Training
In order to train the spectral upsampling model, you can use the following command:

```bash
python3 train.py
```

To change any of the parameters like batch sizes, number of epochs, datasets considered, losses considered etc; check `config.py` file.

If you want to employ Transfer Learning, set `transfer_learning` to True in `config.py`. In order to train on Mobile images, set `use_mobile_dataset` to True.

### Reproduce the results using our pretrained models
To run the inference script, which produces the hypercubes, use the following command:

```bash
python3 inference.py
```

The same parameters apply here. To use a different model type out the name of it in this variable `checkpoint_filename`, and fix the image sizes if they need fixing. The pre-trained models should be in `checkpoints/reconstruction/{others}`.


In order to produce the Mobile Hypercubes, the command is same as above, set `use_mobile_dataset` to True.


**Note**: do not set `use_mobile_dataset` as True, unless you want to perform inference on Mobile Datasets.