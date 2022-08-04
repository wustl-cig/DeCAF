# DECAF
[![DOI](https://zenodo.org/badge/444957070.svg)](https://zenodo.org/badge/latestdoi/444957070)

This is the code repo for the paper ['*Recovery of Continuous 3D Refractive Index Maps from Discrete Intensity-Only Measurements using Neural Fields*'](https://arxiv.org/abs/2112.00002) (previously known as '*Zero-Shot Learning of Continuous 3D Refractive Index Maps from Discrete Intensity-Only Measurements*').

## Download datasets
Available datasets:
- Algae (Figure 2)
- Diatom (aidt, Figure 3)
- Diatom_midt (mdit, Extended Figure 1)
- Cells_b (Figure 4)
- Cells_c (Figure 4)
- Celegans_head (Figure 5)
- Celegans_body/middle (Figure 5)
- Simulated granulocyte cluster (Figure 6)
- Yanny's data (Supplementary)

Data avaliable at https://drive.google.com/drive/folders/1maQxPFHFcouoEFOUo7e5FJSN2hBT78eB?usp=sharing.
Please move data to ```datasets/DATASET_NAME/input/DATASET_NAME.mat```

## Setup the environment
Setup the environment
```
conda env create --file decaf.yml
```
To activate this environment, use
```
conda activate decaf-env
```
To deactivate an active environment, use
```
conda deactivate
```
## Run the code
Run inference:
```
python predict.py --flagfile=datasets/DATASET_NAME/pred_config.txt
```
**NOTE**: We already provide the pre-trained models for each sample in the folder ```datasets/DATASET_NAME/trained_model/```

Run training:
```
python predict.py --flagfile=datasets/DATASET_NAME/train_config.txt
```
**NOTE**: We trained the model on a machine equipped with one AMD Threadripper 3960X 24-core CPU and four Nvidia RTX 3090 GPUs. We parallelized the training of DeCAF over two GPUs to accelerate the convergence. Under this setup, it approximately takes one day to train the model.

Example:
```
python predict.py --flagfile=datasets/Cells_c/pred_config.txt
```

## Expected outputs
After inference, the results will be saved in the folder ```datasets/DATASET_NAME/inference/```, including
```
one data file (.mat)
one video (.mp4)
one stack of images (.tif)
```

## File structure
```
DECAF
  |-datasets
    |-Algae
	  |-inference : inference results (avaliable after running predict.py).
		|- includes .mat, .mp4, and .tif images
	  |-models: training models (avaliable after running main.py).
	  |-input: forward model and measurements (requires additional download).
	  |-trained_model: trained neural representaion weights for the data set.
    |...
  |-model: DECAF model
  |-trained_regularizer: Trained DnCNN denoiser.
  |-main.py : training main function.
  |-predict.py: inference main function.
```
