import streamlit as st
import cv2
import numpy as np
import face_recognition
import pickle
from PIL import Image

# Load pre-trained face encodings model
with open("face_encodings_model1.pkl", "rb") as f:
    data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

# Function to recognize faces and detect landmarks
def recognize_faces_and_landmarks(image, known_encodings, known_names, tolerance=0.4):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    face_landmarks = face_recognition.face_landmarks(image)

    recognized_faces = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        name = "Unknown"
        if distances[best_match_index] < tolerance:
            name = known_names[best_match_index]

        recognized_faces.append((name, (top, right, bottom, left)))

    return recognized_faces, face_landmarks

# Function to annotate faces and landmarks
def annotate_image_with_landmarks(image, recognized_faces, landmarks):
    for name, (top, right, bottom, left) in recognized_faces:
        # Draw rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Add name label
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw landmarks
    for face_landmark in landmarks:
        for key, points in face_landmark.items():
            for point in points:
                cv2.circle(image, point, 2, (255, 0, 0), -1)  # Draw landmarks as blue dots
    return image

# Streamlit app
st.title("Face Recognition with Landmark Detection")

# Menu options
menu = st.sidebar.selectbox(
    "Select Option",
    ["Webcam Face Recognition", "Upload Image Face Recognition"]
)

if menu == "Webcam Face Recognition":
    st.subheader("Real-Time Face Recognition with Landmark Detection")

    if st.button("Start Webcam"):
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.write("Failed to access the webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recognized_faces, landmarks = recognize_faces_and_landmarks(
                rgb_frame, known_encodings, known_names
            )
            annotated_frame = annotate_image_with_landmarks(frame, recognized_faces, landmarks)

            stframe.image(annotated_frame, channels="BGR")

            # Stop webcam when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()

elif menu == "Upload Image Face Recognition":
    st.subheader("Upload an Image for Face Recognition with Landmark Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = np.array(Image.open(uploaded_file))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        recognized_faces, landmarks = recognize_faces_and_landmarks(
            rgb_image, known_encodings, known_names
        )

        # Annotate and display the image
        annotated_image = annotate_image_with_landmarks(image.copy(), recognized_faces, landmarks)
        st.image(annotated_image, caption="Recognized Faces with Landmarks", use_column_width=True)

        # Show results
        if recognized_faces:
            st.write("Detected Faces:")
            for name, _ in recognized_faces:
                st.write(f"- {name}")
        else:
            st.write("No faces recognized.")
