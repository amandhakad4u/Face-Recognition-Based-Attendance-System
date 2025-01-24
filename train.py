import face_recognition
import cv2
import os
import pickle
import numpy as np

# Folder to save captured images for each person
people_images_folder = "people_images/"
os.makedirs(people_images_folder, exist_ok=True)

# Initialize lists to hold encodings and labels
encodings = []
labels = []

def train_model():
    global encodings, labels
    # Loop over all images in the folder and get face encodings
    for image_filename in os.listdir(people_images_folder):
        image_path = os.path.join(people_images_folder, image_filename)
        # Extract the name from the image filename (e.g., 'elon_1.jpg' -> 'elon')
        base_name = os.path.splitext(image_filename)[0]
        name = base_name.split('_')[0]  # Split and take the first part

        # Load the image and find face encodings
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)

        if len(face_encoding) > 0:
            encodings.append(face_encoding[0])  # Use the first face encoding found
            labels.append(name)  # Label with the extracted person's name

    # Convert the encodings and labels to numpy arrays
    encodings = np.array(encodings)
    labels = np.array(labels)

    # Save the trained model (encodings and labels) using pickle in the project folder
    model_filename = "face_recognition_model.pkl"
    project_folder = os.getcwd()  # Get the current project directory
    model_path = os.path.join(project_folder, model_filename)

    with open(model_path, 'wb') as model_file:
        pickle.dump((encodings, labels), model_file)

    print(f"Model saved to {model_path}")



# Train the model using existing images in the folder
train_model()

