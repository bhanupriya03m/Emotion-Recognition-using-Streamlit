import cv2
import streamlit as st
from deepface import DeepFace
import numpy as np

# Function to run emotion detection on the uploaded image
def run_emotion_detection(image_path):
    # Load the image using OpenCV
    frame = cv2.imread(image_path)

    # Predict the emotion, age, and gender from the image with enforce_detection=False
    predictions = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)

    # Get the dominant emotion label and probability
    dominant_emotion = predictions['dominant_emotion']
    emotion_probability = predictions['emotion'][dominant_emotion]

    # Get the predicted age and gender
    age = predictions['age']
    gender = predictions['gender']

    # Draw the emotion label, probability, age, and gender on the image
    emotion_label = f"{dominant_emotion} ({emotion_probability:.2f})"
    cv2.putText(frame, emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    age_gender_label = f"Age: {age}, Gender: {gender}"
    cv2.putText(frame, age_gender_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, dominant_emotion, emotion_probability, age, gender

# Streamlit app code
def main():
    st.title("Emotion Detection from Image")

    # Upload the image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run emotion detection on the uploaded image
        st.text("Running emotion detection...")
        image_path = "temp_image.png"
        frame, dominant_emotion, emotion_probability, age, gender = run_emotion_detection(image_path)
        st.text("Emotion detection complete.")

        # Display the uploaded image with the detected emotion, age, and gender
        st.text("Uploaded Image with Detected Emotion, Age, and Gender:")
        st.image(frame, channels="BGR", use_column_width=True)

        # Display the dominant emotion, probability, age, and gender
        st.text("Dominant Emotion: " + dominant_emotion)
        st.text("Emotion Probability: {:.2f}".format(emotion_probability))
        st.text("Age: " + str(age))
        st.text("Gender: " + gender)

if __name__ == "__main__":
    main()
