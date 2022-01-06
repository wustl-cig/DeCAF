# DECAF
This is the code repo for the paper ['*Zero-Shot Learning of Continuous 3D Refractive Index Maps from Discrete Intensity-Only Measurements*'](https://arxiv.org/abs/2112.00002)

## Download datasets
Available datasets:
- Algae
- Diatom
- Cells_b
- Cells_c
- Celegans_head
- Celegans_middle (body)

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
