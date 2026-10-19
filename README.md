DLE305 - Assessment 2 - Image Segmentation
Rae, Foster

### Overview

This project compares a custom encoder-decoder with a U-Net using a pretrained MobileNetV2 encoder for MRI abnormality segmentation.

Both architectures were trained using Dice, Tversky and focal Tversky loss fucntions. After the best performing loss function was identified further tuning was applied before a final comparison between the two for a final model selection.

### Setup

The only setup besides requirements is ensuring that the curent working directory is the project root for pathing.

### Requirements

Python 3.12 was used for development.

Install the required packages:
`Pip install -r requirements.txt`

Internet connection may be needed the first time MobileNetV2 ImageNet weights are downloaded.

### Project Structure

Keep these files and folders together to run:

    data/
    models/
    history/
    requirements.txt
    DLE305RaeFosterBrief2.ipynb
    segmentation_gui.py

The data folder should contain the patient folders directly. Each patient folder contains the MRI .tif images and their corresponding \_mask.tif files.

The models folder contains saved model checkpoints and initial weights. The history folder contains the training CSV logs.

### Retraining

If you wish to retrain the models on your own local computer, delete the specific .keras file of the model you wish to retrain. Once deleted run the entire notebook again and fresh training for those deleted will begin.

MAKE SURE TO KEEP INITIAL WEIGHT FILES.

### Running GUI

Run `segmentation_gui.py'

1. Click 'Load Image' and select a dataset MRI `.tif` image, not a mask.
2. The matching mask file must be in the folder (it should be already).
3. Once model has loaded, click 'Predict'
4. The original image will be replaced with the coloured overlap overlay:
   - Green: Correctly predicted abnormality.
   - Orange: Missed abnormality.
   - Red: False positive prediction.
5. Load another image to repeat.

The results panel shows Dice, IoU/Jaccard, the percentage of abnormality covered or missed, and the percentage of the predicted area that is incorrect.

Images must be 256x256 (Images in dataset are already) with three channels.

### Final Evaluation Brief

Normal validation data is used for model selection. A fixed augmented validation set is also used to compare performance under different transformations.

The final selected models are:
Custom encoder-decoder trained with Dice loss.
Pretrained U-Net fine tuned with Focal Tversky Loss.

The final evaluation compares these models on the sets set using pixel accuracy, Dice, Jaccard, positive mask Dice and the proportion of empty masks correctly predicted as empty.

The pretrained U-Net ended up being the final selection:

| Model                               | ACC    | Dice   | Jaccard | Positive Dice | Empty masks correct |
| ----------------------------------- | ------ | ------ | ------- | ------------- | ------------------- |
| Fine-tuned pretrained Focal Tversky | 0.9960 | 0.8042 | 0.7713  | 0.6683        | 0.8707              |
