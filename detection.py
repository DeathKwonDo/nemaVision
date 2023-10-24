"""Este código usa o método haarcascade para seguimentar imagens"""

import cv2 as cv


# Load the pre-trained face cascade classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read an image
img = cv.imread('imagem\Opera Snapshot_2022-02-01_234820_meet.google.com.png')

# Convert the image to grayscale for face detection
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the result
cv.imshow('Detected Faces', img)
cv.waitKey(0)
cv.destroyAllWindows()
