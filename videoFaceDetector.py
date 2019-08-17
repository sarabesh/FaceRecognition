import cv2
import model1 as md

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
result = 0
while(result!=1):
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        
        detected_face = cv2.resize(detected_face, (224, 224))
        
        result = md.classify(detected_face)
        print(result)
        result = md.verifyFace('target.jpeg',detected_face)
        cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break   

    
print ('authorized......')
cap.release()
cv2.destroyAllWindows()

