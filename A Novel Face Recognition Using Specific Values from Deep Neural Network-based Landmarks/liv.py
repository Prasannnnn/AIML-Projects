import face_recognition
import cv2
import os
import numpy as np

# Path to the directory with known images
known_faces_dir = r"images"

# Load and encode known faces
known_encodings = []
known_names = []

# Iterate through all files in the known faces directory
for file_name in os.listdir(known_faces_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        # Load the image
        image_path = os.path.join(known_faces_dir, file_name)
        image = face_recognition.load_image_file(image_path)

        # Encode the face
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure at least one face is detected
            known_encodings.append(encodings[0])
            # Extract the name from the filename (e.g., "Alice.png" -> "Alice")
            known_names.append(os.path.splitext(file_name)[0])

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

print("Starting the webcam for real-time face recognition. Press 'q' to quit.")

# Process each frame from the webcam
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate through detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back face locations to original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Calculate distances from known encodings
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        # Determine the name
        tolerance = 0.4  # Adjust as needed (default is 0.6)
        if distances[best_match_index] < tolerance:
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Annotate the name below the face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add confidence score to the annotation
        confidence = 1 - distances[best_match_index]  # Higher confidence is better
        cv2.putText(frame, f"{confidence:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
