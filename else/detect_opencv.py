########################################################################

# OpenCV dir - /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/cv2/
# Video source - https://www.youtube.com/watch?v=PmZ29Vta7Vc

########################################################################

import cv2
import numpy as np

########################################################################

front_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_alt2.xml')
# profile_cascade = cv2.CascadeClassifier('cascade/lbpcascade_profileface.xml')

camera = cv2.VideoCapture(0)

while True:
	# Capture images frame-by-frame.
	ret, frame = camera.read()

	# The cascade only detect gray image.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fronts = front_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	# profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

	for x, y, w, h in fronts:
		# ---------- (frame, upper left, bottom right, color, stroke)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

	# for x, y, w, h in profiles:
	# 	# ---------- (frame, upper left, bottom right, color, stroke)
	# 	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

	# Display the resulting frame.
	cv2.imshow('Facial Recognition', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


# When everything done, release the camera.
camera.release()
cv2.destroyAllWindows()

########################################################################

if __name__ == '__main__':
	pass
