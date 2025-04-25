import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
from PIL import Image  # Add this line to import Image from PIL

# Load pre-trained gender detection model
model = load_model('gender_detection.keras')

# Define the class labels
classes = ['man', 'woman']

st.title("Gender Detection App")

# Option to choose between image upload or webcam detection
option = st.selectbox("Choose Detection Mode", ("Image Upload", "Webcam"))

if option == "Image Upload":
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        image = np.array(Image.open(uploaded_file))

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Detecting...")

        # Apply face detection
        faces, confidence = cv.detect_face(image)

        for idx, f in enumerate(faces):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(image[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.image(image, caption='Processed Image', use_column_width=True)

elif option == "Webcam":
    st.write("Starting webcam...")

    # Open webcam and capture frame using OpenCV
    webcam = cv2.VideoCapture(0)
    frame_window = st.image([])

    while webcam.isOpened():
        status, frame = webcam.read()
        if not status:
            st.write("Failed to open webcam.")
            break

        # Apply face detection
        faces, confidence = cv.detect_face(frame)

        for idx, f in enumerate(faces):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Predict gender
            conf = model.predict(face_crop)[0]
            idx = np.argmax(conf)
            label = classes[idx]
            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame in the Streamlit app
        cv2.imshow("gender detection", frame)

    # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    webcam.release()
