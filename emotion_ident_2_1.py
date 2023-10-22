import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = model_from_json(open("C:/Users/denis/Downloads/model.json", 'r').read())
model.load_weights("C:/Users/denis/Downloads/model2.h5")

cap = cv2.VideoCapture(0)


while True:
    sec, image = cap.read()
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade_db.detectMultiScale(converted_image)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0))

        face_gray = converted_image[y:y+w, x:x+h]
        face_gray = cv2.resize(face_gray, (48, 48))

        image_pixels = np.array(face_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels = image_pixels.reshape(image_pixels.shape[0], 48, 48, 1)

        predictions = model.predict(image_pixels/255)
        max_index = np.argmax(predictions[0])
        emotion_prediction = class_names[max_index]
        text = f'{emotion_prediction} {round(predictions[0][max_index]*100)}%'
        cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
        resize_image = cv2.resize(image, (1080, 720))
        cv2.imshow('model_test', resize_image)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()