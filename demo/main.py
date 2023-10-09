
from datetime import datetime, timedelta
from multiprocessing.sharedctypes import Value
import random
from flask import Flask, render_template, request, redirect, url_for
from mtcnn import MTCNN
import numpy as np
import math, os
import cv2
from tensorflow import keras

#-------------------------------- Global Variables ----------------------------------#

app = Flask(__name__)
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
#detector = MTCNN()  # initialise face detector
model = keras.models.load_model(r'static\IFTADAMAXCP.hdf5')
labels = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
class_labels = {value: key for key, value in labels.items()}

image_path = None
prediction = None

#-------------------------------- Helper Functions ----------------------------------#

# Function to rescale images
def rescale(image):
    size = 128  # set size
    image_size = (size, size, 3)
    height, width, channels = image.shape  # get the shape of image

    #-------------------------------------------#
    # create an empty white image for padding
    img_white = np.ones(image_size, np.uint8)*255
    #-------------------------------------------#
    aspect_ratio = height/width  # get aspect ratio of image

    # If image height is bigger than width, add padding width wise
    if aspect_ratio > 1:
        k = size/height
        wCal = math.ceil(k*width)
        img_resize = cv2.resize(image, (wCal, size))
        imgResizeShape = img_resize.shape
        wGap = math.ceil((size-wCal)/2)
        img_white[:, wGap:wGap+wCal] = img_resize
    # If image width is bigger than height, add padding height wise
    else:
        k = size/width
        hCal = math.ceil(k*height)
        img_resize = cv2.resize(image, (size, hCal))
        imgResizeShape = img_resize.shape
        hGap = math.ceil((size-hCal)/2)
        img_white[hGap:hGap+hCal, :] = img_resize
    return img_white

# Function to preprocess images


def preprocess_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    #global detector
    #faces = detector.detect_faces(img)  # detect faces
    if len(faces) > 0:
        # If a face is detected, crop the image to include only the face
        # ------------------------- take coordinates of edges of the face
        (x, y, width, height) = faces[0]
        # -------------------- crop the face using coordinates
        cropped_image = img[y:y+height, x:x+width]
        # ----------------------- set aspect ratio using the function set previously
        rescaled_image = rescale(cropped_image)
        # - convert to gray scale
        gray_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        # ------------------------- perform gray scale normalization using histogram equalization
        img_eq = cv2.equalizeHist(gray_image)
        return img_eq
    else:
        return None

#-------------------------------- Application Routes ----------------------------------#

@app.route('/result', methods=['GET', 'POST'])
def result():
    global image_path
    global prediction
    return render_template('homepage.html', image_path=image_path, prediction=prediction)
    
@app.route('/', methods=['GET', 'POST'])
def root():
    global image_path
    global prediction
    if request.method == 'GET':
        return render_template('homepage.html')
    else:
        if request.form['submit'] == 'Upload Image':
            # Get the uploaded file from the request
            file = request.files['image']

            # Save the file to a temporary location
            # Specify the path to save the image
            image_path = f'static/files/{file.filename}'
            file.save(image_path)

            img = cv2.imread(image_path)
            img_eq = preprocess_image(img)
            
            if img_eq is not None and len(img_eq) > 0:
                # Convert the resized face image to RGB color format
                rgb_face = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

                # Expand the dimensions to match the expected input shape of the model
                expanded_face = np.expand_dims(rgb_face, axis=0)

                image = expanded_face/255.  # Add this line
                
                # Perform prediction using the loaded model
                prediction = model.predict(image)
                
                # Get the index of the predicted class with the highest probability
                predicted_index = np.argmax(prediction)

                # Retrieve the predicted label based on the index
                prediction = class_labels[predicted_index]
            else:
                prediction = 'Full Face Not Detected'
        return redirect(url_for('result'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
