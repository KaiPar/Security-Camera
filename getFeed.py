import cv2
import numpy as np
from flask import Flask, render_template, Response
import time
import json

face_no = 0

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def capture():
    global face_no
    haar_file = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_file)

    while(True):
        success, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # to convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        face_no = len(faces)
        #print("face no.: ", face_no)
        ret, buffer = cv2.imencode('.jpg', frame)

        for(x, y, w, h) in faces: # To draw boxes around face
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
        
        #print("face no. capture(): ", face_no)
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('index.html', place_name = "Front door", facecount = face_no) # set the place name accoriding to your wish...

@app.route("/get_face_count")
def get_face_count():
    #global face_no
    face_dict = {"facecount":face_no}

    #print ("inside get_face_count: ", face_no)

    return json.dumps(face_dict)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
 