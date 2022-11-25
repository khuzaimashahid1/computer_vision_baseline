# Tensorflow Image Classification Models
This repository contains code for training image classification models using various pre-trained models in Keras. These models can act as baselines for any computer vision task. The models included in this repository are:

VGG16
VGG19
ResNet50
InceptionV3
Xception
InceptionResNetV2
MobileNet
MobileNetV2
DenseNet121
DenseNet169
DenseNet201
NASNetLarge
NASNetMobile
EfficientNet (B0)

# Key features
1 - Last 20 layers of each model are unforzen.
2 - Batch Normalization layers are unfrozen to learn new data statistics.
3 - Image Augmentations are applied
4 - For each model, its own preprocessing function is used so you only need to provide dataset and the script takes care of rest 


# Getting Started

## Prerequisites
Make sure you have the following libraries installed:

TensorFlow 2.x
Keras
EfficientNet
Matplotlib
Numpy

## Usage
You can run the code by executing the train() function in the __main__ block. The train() function takes the following arguments:

modelName: The name of the model to train. Choose from the list of supported models.
batch_size: The batch size for training.
nb_epochs: The number of epochs to train the model.
saveModelJSON (optional): Whether to save the model architecture as a JSON file.
## Example usage:
python train.py modelName batch_size nb_epochs saveModelJSON
If you don't provide any command line arguments, the code will train the MobileNet model with a batch size of 8 and 20 epochs.

## Results
After training, the code will save the trained model weights in a file named modelName.h5 and plot the accuracy and loss history in separate files named modelName_accuracy.png and modelName_Loss_History.png, respectively.

# Future Work
You can add newer models such as RegNet and convNext in a similar way.