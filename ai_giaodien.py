
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

model1 = load_model('C:\Picture\model.h5')

# open webcam
webcam = cv2.VideoCapture(0)

# detectface
face_cascade = cv2.CascadeClassifier('C:\Picture\haarcascade_frontalface_alt.xml')

# class
classes = ['BuonNgu', 'TinhTao']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    flag, frame = webcam.read()

    grayface = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayface)

    for (x,y,w,h) in faces:

        # get corner points of face rectangle        
        # (startX, startY) = f[0], f[1]
        # (endX, endY) = f[2], f[3]
        startX = x
        startY = y
        endX = x + w
        endY = y + h

        # draw rectangle over face
        cv2.rectangle(frame, (startX-10,startY-20), (endX+10,endY+10), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        

        face_crop = np.array(face_crop)
        face_crop_iden = tf.image.resize(face_crop, [150,150])
        face_crop_iden = np.expand_dims(face_crop_iden, axis=0)
        

        # apply drowsy detection on face
        result=(model1.predict(face_crop_iden).argmax())
        # get label with max accuracy
        label = classes[result]
        label = "{}".format(label)
        Y = startY - 20 if startY - 20 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
    # display output
    cv2.imshow("drowsy detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()