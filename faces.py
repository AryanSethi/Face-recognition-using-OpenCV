import numpy as np
import cv2
import pickle


face_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer= cv2.face.LBPHFaceRecognizer_create() #for training
recognizer.read("trainer.yml")

labels={"person_name": 1}
with open('labels.pickle','rb') as f:
    og_labels= pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

webcam= cv2.VideoCapture(0) #0 indicates the index of camera we want to use


while(True):
    ret, frame = webcam.read() #captures frames from vwebcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
    for (x,y,w,h) in faces:
        roi_gray= gray[y:y+h, x:x+w]
        roi_frame = frame[y:y + h, x:x + w]

        id_,conf=recognizer.predict(roi_gray)
        if conf>=45 and conf<=84:
            font= cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke=2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke,cv2.LINE_AA)

        cv2.imwrite("myimage.png", roi_gray)
        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h),color,stroke)

    cv2.imshow('My Experiment',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()