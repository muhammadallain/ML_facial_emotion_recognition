# ML_facial_emotion_recognition

Facial Emotion Recognition model is built using Convolution Neural Networks. Two pretrained models namely Inception V3 and Xception were used for transfer learning while third model was designed from scratch and performance was compared for each model. Dataset images were generated using OpenAI's DALL-E 2 Generative model in bulk and were mixed with two official datasets named KDEF and JAFFE.

## Process
These steps were invloved in the entire process.

### Data Collection
* Generate images using OpenAI's DALL-E 2 for seven emotional expressions.
* Mix images from previous step with two other datasets to obtain master dataset.

### Pre-processing
* Detect faces from images using Multi Task Convolution Neural Network (MTCNN)
* Crop faces from images
* Add padding width wise or height wise to maintain consistant aspect ratio of cropped images.
* Convert Images to grayscale for further consistancy.
* Normalise brightness and contrast ratio for further consistancy.

### Post-processing
Images were manually checked and filtered out in case of bad crops or bad face detection anomalies.

### Training
Training was done with an image size of 128 x 128 for 120 epochs with ADAM optimiser with learning rate of 0.0001.

### Fine Tuning
Fine tuning was done based on different optimisers.

### Testing
Testing was done on unseen data.

### Results
Xception Testing Accuracy = 93.5%

Inception V3 Testing Accuracy = 92.7%

Custom Model Testing Accuracy = 89%
