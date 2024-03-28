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

### Dataset
Download the datasets from the following links:

- Pear Bosc
- Pear Bartlett
- Avocado Organic
- Avocado Hass

The dataset directories are divided as follows:

```
shelflife
│
└── avocado-hass
|	|
|	└── mobile-reconstructed	Inferred Upsampled Mobile Hypercubes 
|	│
|	└── mobile-rgbn			Mobile Image Dataset
|	|
|	└── reconstructed		Inferred Upsampled Hypercubes 
|	|
|	└── rgbn			RGB and NIR images
|	|
|	└── rgbnir-sensor		RGB with IR cutoff removed and NIR
|	|
|	└── split			Training, Validation and Testing split
|
|	496.mat				Ground truth Hypercubes
|	497.mat
|	...
|
└── pear-bartlett
|	|
|	...
```

Some of the directories (mobile-reconstructed, reconstructed, rgbnir-sensor-reconstructed) are created when the `inference.py` script is run.

`split` directory is created when `MobiSLP/dataPreparation/trainTestDiv.py` script is executed.

### Training
In order to train the spectral upsampling model, you can use the following command:

```bash
python3 train.py
```

To change any of the parameters like batch sizes, number of epochs, datasets considered, losses considered etc; check `config.py` file.

If you want to employ Transfer Learning, set `transfer_learning = True` in `config.py`. In order to train on Mobile images, set `use_mobile_dataset = True`.

During training, please also check the `dataset.py` for which input set is loaded into the memory.

### Reproduce the results using our pretrained models
To run the inference script, which produces the hypercubes, use the following command:

```bash
python3 inference.py
```

The same parameters apply here (`config.py`). To use a different model type out the name of it in this variable `checkpoint_filename`, and fix the image sizes if they need fixing. The Pre-trained models can be found here. Place the pre-trained models in `checkpoints/reconstruction/{others}`.

This script also calculates the errors for the six metrics: MRAE, RMSE, SAM, SID, PSNR, SSIM.

In order to produce the Mobile Hypercubes, the command is same as above, set `use_mobile_dataset = True`. The error metrics are not calculated for Mobile dataset, because there are no Ground Truth Hypercubes to compare them to.


**Note**: do not set `use_mobile_dataset = True`, unless you want to perform inference on Mobile Datasets, as it will select wrong band numbers.