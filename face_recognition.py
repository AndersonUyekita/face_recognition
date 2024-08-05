import face_recognition
import cv2
import numpy as np
import os
import re
from PIL import Image

# Função para carregar uma imagem e obter codificações faciais usando PIL
def load_image_and_get_encoding(image_path):
    try:
        # Carregar a imagem usando PIL
        image = Image.open(image_path)
        image = np.array(image)
        # Obter codificações faciais
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            return face_encodings[0]
        else:
            print(f"No face encodings found in image {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Função para carregar imagens de exemplo e aprender a reconhecer elas
def load_known_faces(known_images_folder):
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    for file_name in os.listdir(known_images_folder):
        # Verifica se o arquivo é uma imagem
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Extrai o número de registro (re) e o nome da pessoa
            match = re.match(r'(\d+)_([a-zA-Z_]+)\.jpg', file_name)
            if match:
                re_number, name = match.groups()
                name = name.replace('_', ' ').title()  # Formata o nome
                image_path = os.path.join(known_images_folder, file_name)

                # Carrega a imagem e codifica o rosto
                face_encoding = load_image_and_get_encoding(image_path)
                if face_encoding is not None:  # Verifica se a codificação foi encontrada
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                    known_face_ids.append(re_number)

    return known_face_encodings, known_face_names, known_face_ids


# Caminho completo para a pasta contendo as imagens conhecidas
known_images_folder = r"C:\Users\ander\Documents\jupyter_notebooks\face_recognition\01-data"

# Carregar faces conhecidas
known_face_encodings, known_face_names, known_face_ids = load_known_faces(known_images_folder)

# Verificar se as faces foram carregadas corretamente
if known_face_encodings:
    print("Known faces loaded successfully!")
else:
    print("No known faces loaded.")

# Initialize variables
face_locations = []
face_encodings = []
face_names = []

# Open video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read video frame
    ret, frame = video_capture.read()

    # Ensure the frame was successfully read
    if not ret:
        print("Failed to capture image")
        break

    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the known face name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since we scaled them down
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Add margins to the face locations to include more of the head
        margin = 40
        top = max(0, top - margin)
        right = min(frame.shape[1], right + margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
