# Skidövervakaren
 
 This guide walks you through setting up the Python virtual environment and installing dependencies.

## Setting Up the Virtual Environment

**Ensure Python is Installed**  
Check if Python is installed by running:

```
python --version
```
or

```
python3 --version
```

> [!IMPORTANT]
> This program requires TensorFlow that can only be used with Python 3.9-3.12
> Make sure you have a compatible Python version before installing requirements

### Create a Virtual Environment

Run the following command in your project directory:

``` 
python -m venv venv 
```

or

```
python3 -m venv venv 
```

### Activate the Virtual Environment

Windows (Command Prompt / PowerShell)

```
venv\Scripts\activate
```

Mac/Linux

```
source venv/bin/activate
```
Once activated, you should see (venv) in your terminal.

### Installing Dependencies

After activating the virtual environment, install required dependencies with:

```
pip install -r requirements.txt
```
To add new dependencies and update requirements.txt, run:

```
pip freeze > requirements.txt
```
> [!IMPORTANT]
> If installation of requirements fails, make sure you have compatible Python version and latest version of pip

### Deactivating the Environment

When finished, deactivate the virtual environment by running:

```
deactivate
```

## Running the Model

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and EfficientNetB0 to classify images of snowboarders and skiers.

### Dataset Structure
Ensure your dataset is structured as follows:
```
Data/CNN/
    train/
    val/
    test/
```
Each folder should contain images categorized into respective classes.

### Training the Model

The model is trained in two phases:
1. **Initial Training:** The EfficientNetB0 base model is frozen, and only the classifier layers are trained.
2. **Fine-tuning:** The base model is unfrozen and trained on the dataset with a lower learning rate.

### Evaluating the Model
After training, the model is evaluated on the test dataset:

The test accuracy will be displayed in the terminal.

### Saving the Model
The trained model is saved in the `cnn_classifier.keras` file and can be loaded for inference.

## Notes
- The model processes images with a target size of **224x224**.
- Data augmentation (rescaling, rotation, zoom, flipping, standardization) is applied.
- Uses **Adam optimizer** with fine-tuning capabilities.
- Default training runs for **15 epochs** initially, then **10 epochs** for fine-tuning.