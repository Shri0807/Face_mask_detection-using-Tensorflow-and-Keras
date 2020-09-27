# Face mask Detection using TensorFlow, Keras and OpenCV

This project is a face mask classifier used to identify wheter a person is wearing a facemask or not. The Neural Network was built using TensorFlow and Keras.

Main Libraries used in this Project:-
* TensorFlow 2.3.0
* Keras 2.3.1
* OpenCV 4.2.32

Other Libraries include Numpy, scikit-learn and Matplotlib.

The Project has 2 Python files:
* Model_train.py
* mask_video.py

--------------------------------------------------------

## Model_train.py

This python file is used to train the Deep Learning Neural Network. The Images are loaded into the files from the dataset folder. Each Image is preprocessed and converted into a Numpy array. The images are normalized and augmented using ImageDataGenerator() function of Keras. 

Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It helps reduce overfitting when training a machine learning. It is clesely related to oversampling in data analysis.

The trained model is saved as mask_detector.model using Keras. The Model is built over MobileNet_v2. MobileNet-v2 is a convolutional neural network that is 53 layers deep. I added additonal layers on top of MobileNet_v2 to achieve better accuracy. 

Note:-
> The only Prerequisite for MobileNet_v2 is that the image needs to of size 224x224. Remember to resize the image when using MobileNet_v2.

The trained model is later loaded into mask_video.py for testing.

--------------------------------------------------------

## mask_video.py

This file uses OpenCV and Numpy. The Trained Model is loaded using Keras load_model() function. Using cv2.VideoCapture() function the video from webcam is captured. The python script also uses another trained Face detector for detecting faces from Webcam Video feed. 

First a face is detected and on the detected face the facemask detector is applied to check for presence of mask. 

Note:-
> You can also use haarcascade classifier provided my OpenCV for detecting a Face, but the accuracy provided by it is very low.
-------------------------------------------------------

The Output folder has the output for With mask and without mask.

# How to run the Project

1. Create a python virtual environment and install the dependencies.
2. Run the Model_train.py file for training the Model.
3. Once training is completed Run mask_video.py for testing the Model.
