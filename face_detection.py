import cv2
import time
import pygame

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier("C:\\Users\\Harshith Y\\Downloads\\haarcascade_frontalface_default.xml")

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

# Initialize variables for tracking time
last_detected_time = time.time()
alert_threshold = 5  # Threshold for alert in seconds

# Function to play buzzer sound

def play_buzzer_sound():
    pygame.mixer.music.load("C:\\Users\\Harshith Y\\Downloads\\buzzer_sound.wav")
    pygame.mixer.music.play()

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Check if face is detected
    if len(faces) > 0:
        # Update last detected time
        last_detected_time = time.time()
        # Display activity in terminal
        print("User is looking at the screen:", time.strftime("%H:%M:%S", time.localtime()))

    # Check if face is not detected for more than threshold seconds
    if time.time() - last_detected_time > alert_threshold:
        # Display activity in terminal
        print("User is away from the screen:", time.strftime("%H:%M:%S", time.localtime()))
        cv2.putText(img, "Alert: User is away from the screen!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Play buzzer sound
        play_buzzer_sound()

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
