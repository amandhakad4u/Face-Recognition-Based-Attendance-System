
import face_recognition
import cv2
import os
import pickle
import numpy as npq
import datetime
import pandas as pd
# Folder to save captured images for each person
today_date = datetime.datetime.now().strftime('%Y-%m-%d')

def recognize_face():
    # Load the trained face recognition model (encodings and labels)
    model_filename = "face_recognition_model.pkl"
    project_folder = os.getcwd()  # Get the current project directory
    model_path = os.path.join(project_folder, model_filename)

    with open(model_path, 'rb') as model_file:
        encodings, labels = pickle.load(model_file)

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)  # Use the default webcam

    print("Press 'q' to quit the recognition program.")

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        # Find all faces in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through all faces found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the face with the saved encodings
            matches = face_recognition.compare_faces(encodings, face_encoding)

            name = "Unknown"  # Default if no match is found
            rectangle_color = (0, 0, 255)  # Default rectangle color (red)





            # If a match is found, assign the corresponding name and change rectangle color to green
            if True in matches:
                first_match_index = matches.index(True)
                name = labels[first_match_index]
                rectangle_color = (0, 255, 0)  # Change rectangle color to green if match is found
            try:
                df = pd.read_csv('attend.csv', index_col=0)
    
            except FileNotFoundError:
          
                df = pd.DataFrame(index=["elon", "bill", "robert"])



            # Check if today's date column exists, if not, add it
            if today_date not in df.columns:
                df[today_date] = None  # Add new column with empty values

            # Add a value in the 'elon' row under today's date column
            df.at[name, today_date] = "Present"  # Replace "Aman Dhakad" with your desired value

            # Save the updated DataFrame back to the CSV file
            df.to_csv('attend.csv')




            # Draw a rectangle around the face and put the name label
            cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, rectangle_color, 1)
            
            













        # Display the resulting image
        cv2.imshow('Face Recognition', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

recognize_face()