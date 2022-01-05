# DECAF

Available datasets:
- Algae
- Diatom
- Cells_b
- Cells_c
- Celegans_head
- Celegans_middle (body)

Data avaliable at https://drive.google.com/file/d/16XMJvbrAVacskfLDNGQxnBplLjvAL-7D/view?usp=sharing
Please move data to ```datasets/DATASET_NAME/input/DATASET_NAME.mat```

Run inference:
```
python predict.py --flagfile=datasets/DATASET_NAME/pred_config.txt
```

Run training:
```
python predict.py --flagfile=datasets/DATASET_NAME/train_config.txt
```

Example:
```
python predict.py --flagfile=datasets/Celegans_middle/pred_config.txt
```

File structure:
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
