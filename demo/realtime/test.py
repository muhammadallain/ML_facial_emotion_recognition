import cv2, math, os
import numpy as np
from tensorflow import keras
from mtcnn import MTCNN

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(haar_model)

# Load your trained emotion detection model
model = keras.models.load_model('realtime/inception.hdf5')

# Define the emotions corresponding to the model's output classes
#emotions = ['anger','disgust','fear', 'happiness', 'neutral', 'sadness', 'surprise']
emotions = ['anger', 'happiness', 'neutral']


detector = MTCNN() # initialise face detector

# Function to preprocess images
def preprocess_image(img):
    #global detector
    #faces = detector.detect_faces(img) # detect faces
    #if len(faces) > 0:
        # If a face is detected, crop the image to include only the face
    rescaled_image = rescale(img) #----------------------- set aspect ratio using the function set previously
    gray_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY) #- convert to gray scale
    img_eq = cv2.equalizeHist(gray_image) #------------------------- perform gray scale normalization using histogram equalization
    return img_eq



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

# Function to predict emotion from face image
def predict_emotion(face_image):
    # Preprocess the face image (resize, normalize, etc.) if needed
    # ...

    # Perform emotion prediction using the loaded model
    prediction = model.predict(face_image)
    print(prediction)
    predicted_emotion = emotions[np.argmax(prediction)]

    return predicted_emotion

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    p = 20 # padding
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the face region of interest
        face_roi = frame[y+p:y+h+p, x+p:x+w+p]

        # Preprocess the face image if needed
        # ...

        # Convert the face image to the format expected by the model (e.g., resize, normalize, etc.)
        face_input = preprocess_image(face_roi)
        
        # Convert the resized face image to RGB color format
        rgb_face = cv2.cvtColor(face_input, cv2.COLOR_GRAY2RGB)

        # Expand the dimensions to match the expected input shape of the model
        expanded_face = np.expand_dims(rgb_face, axis=0)

        image = expanded_face/255.  # Add this line
        # Predict the emotion from the face image
        emotion = predict_emotion(image)

        # Display the predicted emotion on the frame
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
