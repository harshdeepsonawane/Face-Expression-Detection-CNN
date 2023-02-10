import cv2
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import pandas as pd



# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
# json_file = open('emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # # load weights into new model
# emotion_model.load_weights("model_weight.h5")
# print("Loaded model from disk")

# start the webcam feed
model = load_model("model_grey.h5")
Face_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    face_rect = face_detector.detectMultiScale(gray_img,scaleFactor=1.1, minNeighbors=9)
   
    for (x, y, w, h) in face_rect:
        rec = cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_img[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(29,29)),-1),0)
        emotion_prediction = model.predict(cropped_img)
        val = []
        for i in emotion_prediction:
            val.append(i)
        data_fram = pd.DataFrame([Face_name,val[0]]).T
        data_fram.columns = ["Face_Name","Output"]

        # Final Output
        op_final = data_fram.groupby('Output').max().tail(1).values[0][0]

        cv2.putText(frame,op_final,(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()