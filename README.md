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

## Running the CNN Model

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

## PyTorch + CUDA + cuDNN Setup and Verification

### **Setup Summary**
This part outlines the setup, installation, and verification steps to ensure **PyTorch** is correctly using **CUDA (GPU acceleration)** and **cuDNN** for deep learning operations.

### **Check CUDA Installation**
Verify that CUDA is installed and available on your system:
```
nvcc --version
```
Expected output (example):
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed Sep 21 10:41:10 Pacific Daylight Time 2022
Cuda compilation tools, release 11.8, V11.8.89
```

### **Check GPU Availability**
Confirm that your GPU is recognized using:
```
nvidia-smi
```
Expected output should display **your GPU model** (e.g., `NVIDIA GeForce RTX 2080 SUPER`) and **CUDA version**.

### **Verify PyTorch and CUDA Compatibility**
Run the following inside Python to check if PyTorch detects CUDA:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("PyTorch CUDA Version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0))
```
Expected output (example):
```
CUDA Available: True
PyTorch CUDA Version: 12.1
Number of GPUs: 1
GPU Name: NVIDIA GeForce RTX 2080 SUPER
```

> [!IMPORTANT]
> Make sure you have the correct matching versions of cuda. Otherwise it will not work properly

### **4. Verify cuDNN Installation**
Run the following to check if cuDNN is enabled and its version:
```python
import torch
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
print("cuDNN Version:", torch.backends.cudnn.version())
```
Expected output (example):
```
cuDNN Enabled: True
cuDNN Version: 90100
```

## Dataset Preparation and YOLO Training

### Dataset Splitting and Organization

The dataset is automatically split into train (80%), validation (10%), and test (10%) sets. The script ensures that both images and their corresponding annotation files (.txt) are placed correctly into their respective directories.

Directory Structure after Split:
```
Data/
 ├── split_dataset/
 │   ├── train/
 │   │   ├── images/
 │   │   ├── labels/
 │   ├── val/
 │   │   ├── images/
 │   │   ├── labels/
 │   ├── test/
 │   │   ├── images/
 │   │   ├── labels/
 │   ├── classes.txt
 │   ├── mydataset.yml
 ```

images/ contains all image files.
labels/ contains YOLO annotation files.
classes.txt holds the list of class names.

### Generating mydataset.yml

The dataset_builder script automatically generates a mydataset.yml configuration file, which is required for training with YOLO. The YAML file contains:

The dataset root directory (path),
the subdirectories for training, validation, and test data
The number of classes (nc)
and the class names

Example mydataset.yml:

```bash
#mydataset.yml - YOLO dataset configuration

path: C:/Users/YourUser/Desktop/Project/Data/split_dataset
train: train/images
val: val/images
test: test/images

#Number of classes
nc: 2

#Class names
names:
  0: skier
  1: snowboarder
```

> [!IMPORTANT]
> In dataset_builder scripts, insert the full path to ensure that YOLO will find the path correctly.

### Running YOLO Training

To start training the YOLOv11 model, run the following script:

```python
from ultralytics import YOLO

#Load the model
model = YOLO('yolov11n.pt')

#Train the model
model.train(
    data='C:/Users/YourUser/Desktop/Project/Data/split_dataset/mydataset.yml',
    epochs=500,
    patience=0,
    batch=-1,
    project='./Results',
    device='cuda',  # Change to 'cpu' if no GPU is available
    workers=4
)
```

### Model Output

After training is complete, the trained model will be saved in:

```
Results/
 ├── train/
 │   ├── weights/
 │   │   ├── best.pt
 │   │   ├── last.pt
```

best.pt - Best performing model checkpoint

last.pt - Last training epoch model checkpoint

To use the trained model for inference:

```python
model = YOLO('Results/train/weights/best.pt')
results = model.predict('test_image.jpg')
```

> [!NOTE]
> Running a vast amount of epochs takes a long time. GPU usage is recommended as it goes much faster