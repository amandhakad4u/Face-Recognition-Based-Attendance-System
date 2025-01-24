import face_recognition
import cv2
import os

# Folder to save captured images for each person
people_images_folder = "people_images/"
os.makedirs(people_images_folder, exist_ok=True)

# Prompt for person's name to be used in the training data
person_name = input("Enter the name of the person for training: ")

# Set the number of images to capture
num_images = 10

# Initialize the webcam
video_capture = cv2.VideoCapture(0)  # Use the default webcam

print(f"Capturing {num_images} images for {person_name}. Press 'c' to capture an image.")

captured_images = 0
while captured_images < num_images:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(frame)

    # Draw rectangles around the faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        
        cv2.rectangle(frame, (left-100, top-100), (right+100, bottom+100), (0, 255, 0), 2)

    # Display the frame with face detection
    cv2.imshow('Video', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # If 'c' key is pressed, capture and save the image
        if len(face_locations) > 0:  # Ensure a face is detected before saving
            captured_images += 1
            image_filename = os.path.join(people_images_folder, f"{person_name}_{captured_images}.jpg")
            cv2.imwrite(image_filename, frame)  # Save the entire frame (no cropping)
            print(f"Image {captured_images} captured and saved as {image_filename}")
        else:
            print("No face detected. Try again.")

    if key == ord('q'):  # If 'q' key is pressed, quit the program
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
