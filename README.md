# MACHINE LEARNING KAGGLE IN-CLASS COMPETITION: Image Classification on CIFAR-10 Dataset

The CIFAR-10 data consists of 50,000 32x32 color images in 10 classes. There are 40,000 training images and 10,000 test images in the data. We have preserved the train/test split from the original dataset.

The label classes in the dataset are:

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## Implementation and Working

Developed a model to classify 50000, 32x32 color images in 10 classes. It uses image augmentation techniques such as:
1. featurewise_center=False,  
2. samplewise_center=False, 
3. featurewise_std_normalization=False,  
4. samplewise_std_normalization=False,  
5. zca_whitening=False,  
6. rotation_range=10,  
7. width_shift_range=0.1,  
8. height_shift_range=0.1, 
9. horizontal_flip=True, 
10. vertical_flip=False

Convolutional neural network architecture was designed and implemented from scratch. It consists of 14 convolutions and max pool layers with batch normalization and dropout. It consists of two dense layers. The activation functions used are 'elu' and 'softmax', and uses Adam optimizer.

Achieved the following accuracies:

1. Train accuracy: 0.9746
2. Validation accuracy: 0.9145
3. Test accuracy: 0.9210

## Startup:
1. Download the .ipynb file.
2. Open the file in Google Colab.
3. Download the kaggle .json file (API KEY) and upload it in *cell 2*. 
4. Replace the command: "kaggle competitions download -c cmpe-257-lab-2-part-2" by "kaggle competitions download -c cifar-10".

## Requirements:
1. Python 3.7 or higher.
2. Tensorflow 2 or higher.
3. pip version 19.0 or higher.
4. GPU: NVIDIA Tesla K80 or higher.

## Note:
1. Takes time to train the model.
2. To know more about the dataset: https://www.cs.toronto.edu/~kriz/cifar.html


