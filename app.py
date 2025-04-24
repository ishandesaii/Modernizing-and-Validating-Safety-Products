#importing all required libraries
import os
import re
import glob
import yaml
import cv2
import torch
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set3')
import IPython.display as display
from IPython.display import Video
from PIL import Image
import mysql.connector
import face_recognition
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from datetime import timedelta, date, datetime
from flask import Flask, request, render_template, redirect, url_for, session

shutil.rmtree(r'./runs', ignore_errors=True)

#path
fireinputvideopath = './uploadedvideo/finput_video.mp4'
safetyinputvideopath = './uploadedvideo/input_video.mp4'

app = Flask(__name__)
app.secret_key = 'your secret key'

#mysql(xamppserver)
#database connection
mydb = mysql.connector.connect(
  host="localhost",
  port=3306,
  user="root",
  password="",
  database='smart_surveillance_db'
)
print(mydb)
mycursor = mydb.cursor()

# mycursor.execute("truncate FireSmoke_Log")

# mycursor.execute("CREATE TABLE IF NOT EXISTS FireSmoke_Log (fs_id int(11) NOT NULL AUTO_INCREMENT, DateTime DATETIME, frame VARCHAR(50), class_name VARCHAR(50), class VARCHAR(50), confidence VARCHAR(50), x_center VARCHAR(50), y_center VARCHAR(50), width VARCHAR(50), height VARCHAR(50), x_min VARCHAR(50), x_max VARCHAR(50),  y_min VARCHAR(50), y_max VARCHAR(50), width_px VARCHAR(50), height_px VARCHAR(50), PRIMARY KEY (fs_id))")
# mycursor.execute("CREATE TABLE IF NOT EXISTS Safety_Log (s_id int(11) NOT NULL AUTO_INCREMENT, DateTime DATETIME, frame VARCHAR(50), class_name VARCHAR(50), class VARCHAR(50), confidence VARCHAR(50), x_center VARCHAR(50), y_center VARCHAR(50), width VARCHAR(50), height VARCHAR(50), x_min VARCHAR(50), x_max VARCHAR(50),  y_min VARCHAR(50), y_max VARCHAR(50), width_px VARCHAR(50), height_px VARCHAR(50), PRIMARY KEY (s_id))")
# mycursor.execute("CREATE TABLE IF NOT EXISTS Trespass_Log (tp_id int(11) NOT NULL AUTO_INCREMENT, DateTime DATETIME, class_name VARCHAR(50), PRIMARY KEY (tp_id))")


#functions

def countfunc(mycursor):
    sql="SELECT COUNT(*) AS 'COUNT' FROM firesmoke_log WHERE class_name = 'fire' OR class_name = 'smoke'"
    mycursor.execute(sql,)
    fs_count = mycursor.fetchone()

    sql="SELECT COUNT(class_name) AS 'COUNT' FROM safety_log"
    mycursor.execute(sql,)
    s_count = mycursor.fetchone()

    sql="SELECT COUNT(*) AS 'COUNT' FROM trespass_log WHERE class_name = 'Unknown'"
    mycursor.execute(sql,)
    tp_count = mycursor.fetchone()

    return [fs_count[0], s_count[0], tp_count[0]]

#safetygeardetection
def safetygeardetect(inputvidpath):
    class CFG:
        ### inference: use any pretrained or custom model
        # WEIGHTS = 'yolov8x.pt' # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        WEIGHTS = './model/best.pt'
        
        CONFIDENCE = 0.55 # 0.35
        CONFIDENCE_INT = int( round(CONFIDENCE * 100, 0) )
        
        CLASSES_TO_DETECT = [0, 2, 4, 5, 7] # Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest
        
        VERTICES_POLYGON = np.array([[200,720], [0,700], [500,620], [990,690], [820,720]])

        EXP_NAME = 'ppe'

        ### just some video examples
        VID_001 = inputvidpath

        ### choose filepath to make inference on (image or video)
        PATH_TO_INFER_ON = VID_001
        EXT = PATH_TO_INFER_ON.split('.')[-1] # get file extension
        FILENAME_TO_INFER_ON = PATH_TO_INFER_ON.split('/')[-1].split('.')[0] # get filename

        ### paths
        ROOT_DIR = './uploadedvideo/'
        OUTPUT_DIR = './'

    glob.glob(CFG.ROOT_DIR + '*')

    def get_video_properties(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        properties = {
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int( cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) ),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        # Release the video capture object
        cap.release()

        return properties
    
    video_properties = get_video_properties(CFG.VID_001)
    video_properties


    model = YOLO(CFG.WEIGHTS)



    results = model.predict(
        source = CFG.PATH_TO_INFER_ON,
        save = True,
        classes = CFG.CLASSES_TO_DETECT,
        conf = CFG.CONFIDENCE,
        save_txt = True,
        save_conf = True,
        show = True,   #comment this to not show the screen
        device = 'cpu',
    #     stream = True,
    )

    RAW_INFERENCE_VIDEO = glob.glob('./uploadedvideo/input_video*')[0] # avi or mp4
    OUT_VIDEO_NAME = './input_video.mp4'
    print(OUT_VIDEO_NAME)
    print(RAW_INFERENCE_VIDEO)
    subprocess.run(
        [
            "ffmpeg",  "-i", RAW_INFERENCE_VIDEO, "-crf",
            "18", "-preset", "veryfast", "-hide_banner", "-loglevel",
            "error", "-vcodec", "libx264", OUT_VIDEO_NAME
        ]
    )

    Video(data=OUT_VIDEO_NAME, embed=True, height=int(video_properties['height'] * 0.5), width=int(video_properties['width'] * 0.5))

    raw_inference_video_properties = get_video_properties(RAW_INFERENCE_VIDEO)
    raw_inference_video_properties


    def get_predictions_df_for_videos(run = '', model = model, EXP_NAME = CFG.EXP_NAME, video_info = video_properties, save_df = True):
        df = pd.DataFrame()

        root = f'./runs/detect/predict{run}/labels/'
        pieces = sorted([i for i in os.listdir(root)])

        ### iterate over txt files (there is one for each frame in the video)
        for i, frame_txt in enumerate([i for i in pieces]):

            df = pd.read_csv(root + frame_txt, sep=" ", header=None) # read txt file as dataframe
            df.columns = ["class", "x_center", "y_center", "width", "height", "confidence"] # name columns (detection task)

            ### create column 'frame'
            frame_number = re.findall(r'\d+', frame_txt)[-1] # find frame in each txt filename
            df['frame'] = [int(frame_number) for i in range(len(df))]

            if i == 0:
                df_concat = df
            else:
                df_concat = pd.concat([df_concat, df], axis=0).reset_index(drop=True)

        ### create 4 new columns (coordinates converted into pixels): Calculate bounding box coordinates
        df_concat['x_min'] = (df_concat['x_center'] * video_info['width']) - ((df_concat['width'] * video_info['width'])/2)
        df_concat['x_max'] = (df_concat['x_center'] * video_info['width']) + ((df_concat['width'] * video_info['width'])/2)
        df_concat['y_min'] = (df_concat['y_center'] * video_info['height']) - ((df_concat['height'] * video_info['height'])/2)
        df_concat['y_max'] = (df_concat['y_center'] * video_info['height']) + ((df_concat['height'] * video_info['height']/2))

        ### create 2 new columns: height and width in pixels: will be used to filter out bad predictions (false positives)
        df_concat['width_px'] = (df_concat['width'] * video_info['width']).round(0).astype(int)
        df_concat['height_px'] = (df_concat['height'] * video_info['height']).round(0).astype(int)

        ### sort by frame
        df_concat = df_concat.sort_values(by='frame', ascending=True).reset_index(drop=True)

        ### add column 'class_name' and rearrange order
        df_concat['class_name'] = df_concat['class'].map(model.names)
        other_cols = [col for col in df_concat.columns.to_list() if col not in ['frame', 'class_name', 'class', 'confidence' ]]
        new_order = ['frame', 'class_name', 'class', 'confidence'] + other_cols
        df_concat = df_concat[new_order]

        ### export detections df
        if save_df:
            df_concat.to_csv('./static/img/outputlogs/sresult.csv', index=False)

        return df_concat


    out_df = get_predictions_df_for_videos(
        run = '',
        model = model,
        EXP_NAME = CFG.EXP_NAME,
        video_info = video_properties,
        save_df = True
    )

    #remove old account directory
    shutil.rmtree(r'./runs', ignore_errors=True)

    return out_df


#fireandsmokedetection
def firesmokedetect(inputvidpath):

    def get_video_properties(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        properties = {
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int( cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) ),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        # Release the video capture object
        cap.release()

        return properties
    
    video_properties = get_video_properties(inputvidpath)
    video_properties

    model = YOLO( "./model/firebest.pt" )

    results = model( source = inputvidpath, stream = True,  show = True, save = True, save_txt = True, save_conf = True, conf = 0.5 )
    
    for r in results:
        try:    
            # detected fire confidence bigger than 0.8
            if r.boxes.conf[0] > torch.tensor([0.8]).cuda() :
                print("\nWarning Fire\n")
                p1 = Process()
                p1.start()
                break
            # detected fire and smoke at the same time
            elif ( torch.tensor([0.]).cuda() in r.boxes.cls ) and ( torch.tensor([1.]).cuda() in r.boxes.cls ) :
                print("\nWarning Fire\n")
                p1 = Process()
                p1.start()
                break
            # predicted, but no match confidece or class name
            else :
                #print("\nnothing\n")
                pass
        except :  # no predicted
            #print("\nnothing\n")
            pass

    #dataframe
    video_info = video_properties
    save_df = True
    df = pd.DataFrame()
    run = ''
    root = f'./runs/detect/predict{run}/labels/'
    pieces = sorted([i for i in os.listdir(root)])

    ### iterate over txt files (there is one for each frame in the video)
    for i, frame_txt in enumerate([i for i in pieces]):

        df = pd.read_csv(root + frame_txt, sep=" ", header=None) # read txt file as dataframe
        df.columns = ["class", "x_center", "y_center", "width", "height", "confidence"] # name columns (detection task)

        ### create column 'frame'
        frame_number = re.findall(r'\d+', frame_txt)[-1] # find frame in each txt filename
        df['frame'] = [int(frame_number) for i in range(len(df))]

        if i == 0:
            df_concat = df
        else:
            df_concat = pd.concat([df_concat, df], axis=0).reset_index(drop=True)

    ### create 4 new columns (coordinates converted into pixels): Calculate bounding box coordinates
    df_concat['x_min'] = (df_concat['x_center'] * video_info['width']) - ((df_concat['width'] * video_info['width'])/2)
    df_concat['x_max'] = (df_concat['x_center'] * video_info['width']) + ((df_concat['width'] * video_info['width'])/2)
    df_concat['y_min'] = (df_concat['y_center'] * video_info['height']) - ((df_concat['height'] * video_info['height'])/2)
    df_concat['y_max'] = (df_concat['y_center'] * video_info['height']) + ((df_concat['height'] * video_info['height']/2))

    ### create 2 new columns: height and width in pixels: will be used to filter out bad predictions (false positives)
    df_concat['width_px'] = (df_concat['width'] * video_info['width']).round(0).astype(int)
    df_concat['height_px'] = (df_concat['height'] * video_info['height']).round(0).astype(int)

    ### sort by frame
    df_concat = df_concat.sort_values(by='frame', ascending=True).reset_index(drop=True)

    ### add column 'class_name' and rearrange order
    df_concat['class_name'] = df_concat['class'].map(model.names)
    other_cols = [col for col in df_concat.columns.to_list() if col not in ['frame', 'class_name', 'class', 'confidence' ]]
    new_order = ['frame', 'class_name', 'class', 'confidence'] + other_cols
    df_concat = df_concat[new_order]

    ### export detections df
    if save_df:
        df_concat.to_csv('./static/img/outputlogs/fsresult.csv', index=False)
    
    #remove old account directory
    shutil.rmtree(r'./runs', ignore_errors=True)

    return df_concat


#trespassing
def TrackImages():
    class_name=[]
    path = './static/img/Images'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cls in myList:
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        classNames.append('Known')
    print(classNames)

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList


    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    camera = cv2.VideoCapture(0)

    while True:

        ret, frame = camera.read()

        if frame is None:
            break

        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        
        if len(facesCurFrame) != 0 and len(encodesCurFrame) != 0:

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    print(faceDis)
                    matchIndex = np.argmin(faceDis)

                    if faceDis[matchIndex] < 0.50:
                        name = 'Known'

                    else:
                        name = 'Unknown'
                       
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    class_name.append(name)
                    cv2.imshow("WebCam", frame)
                    if name == 'Unknown':
                        frame1=frame
                        leng = len(os.listdir('./static/img/trespassimg'))+1
                        img_name = os.path.join('./static/img/trespassimg/',"img_"+str(leng)+".jpg")
                        cv2.imwrite(img_name, frame1)
                        
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key % 256 == 8:
                break

        else:
            name = 'No Face Detected!!'
            cv2.putText(frame, name, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            class_name.append(name)
            cv2.imshow("WebCam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key % 256 == 8:
                break
        
    camera.release()
    cv2.destroyAllWindows()

    return class_name


#flask
@app.route('/')
def home():
    cntlist = countfunc(mycursor)
    return render_template('index.html',cntlist=cntlist)


@app.route('/index')
def index():
    cntlist = countfunc(mycursor)
    return render_template('index.html',cntlist=cntlist)


@app.route('/firedetectionservice')
def firedetectionservice():
    outputvalues = "nonoutput"
    return render_template('firedetectionservice.html', outputvalues=outputvalues)


@app.route('/safetygeardetectionservice')
def safetygeardetectionservice():
    outputvalues = "nonoutput"
    return render_template('safetygeardetectionservice.html', outputvalues=outputvalues)


@app.route('/usertrespassingservice')
def usertrespassingservice():
    outputvalues = "nonoutput"
    return render_template('trespassingservice.html', outputvalues=outputvalues)


@app.route('/firesmokedetection', methods=['POST'])
def firesmokedetection():
    if os.path.exists('./uploadedvideo/input_video.mp4'):
        os.remove('./uploadedvideo/input_video.mp4')
    file = None
    file = request.files['file']
    file.save(fireinputvideopath)
    inputvidpath = fireinputvideopath
    df_concat = firesmokedetect(inputvidpath)
    for i,row in df_concat.iterrows():
        sql="INSERT INTO FireSmoke_Log VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s)"
        mycursor.execute(sql, (datetime.now(), row['frame'], row['class_name'], row['class'], row['confidence'], row['x_center'], row['y_center'], row['width'], row['height'], row['x_min'], row['x_max'], row['y_min'], row['y_max'], row['width_px'], row['height_px'],))
        mydb.commit()
    os.remove('./static/img/outputlogs/fsresult.csv')
    return render_template('firedetectionservice.html')


@app.route('/safetygeardetection', methods=['POST'])
def safetygeardetection():
    if os.path.exists('./uploadedvideo/finput_video.mp4'):
        os.remove('./uploadedvideo/finput_video.mp4')
    file = None
    file = request.files['file']
    file.save(safetyinputvideopath)
    inputvidpath = safetyinputvideopath
    df_concat = safetygeardetect(inputvidpath)
    for i,row in df_concat.iterrows():
        sql="INSERT INTO Safety_Log VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s)"
        mycursor.execute(sql, (datetime.now(), row['frame'], row['class_name'], row['class'], row['confidence'], row['x_center'], row['y_center'], row['width'], row['height'], row['x_min'], row['x_max'], row['y_min'], row['y_max'], row['width_px'], row['height_px'],))
        mydb.commit()
    os.remove('./static/img/outputlogs/sresult.csv')
    os.remove('./input_video.mp4')
    return render_template('safetygeardetectionservice.html')


@app.route('/trespassdetection', methods=['POST'])
def trespassdetection():
    class_name = TrackImages()
    for i in class_name:
        sql="INSERT INTO Trespass_Log VALUES (NULL, %s, %s)"
        mycursor.execute(sql, (datetime.now(), i,))
        mydb.commit()
    return render_template('trespassingservice.html')


@app.route('/viewfiresmokelogs', methods =['GET'])
def viewfiresmokelogs():
    mycursor = mydb.cursor(dictionary=True)
    sql="SELECT DateTime,class_name,confidence FROM `firesmoke_log` "
    mycursor.execute(sql,)
    logsdatabase = mycursor.fetchall()

    return render_template('viewlogs.html', logsdatabase=logsdatabase)


@app.route('/viewsafetylogs', methods =['GET'])
def viewsafetylogs():
    mycursor = mydb.cursor(dictionary=True)
    sql="SELECT DateTime,class_name,confidence FROM `safety_log` "
    mycursor.execute(sql,)
    logsdatabase = mycursor.fetchall()

    return render_template('viewlogs.html', logsdatabase=logsdatabase)


@app.route('/trespasslogs', methods =['GET'])
def trespasslogs():
    mycursor = mydb.cursor(dictionary=True)
    sql="SELECT DateTime,class_name FROM `trespass_log` "
    mycursor.execute(sql,)
    logsdatabase = mycursor.fetchall()

    return render_template('viewtreslogs.html', logsdatabase=logsdatabase)


if __name__ == '__main__':
    # Run the application
    app.run(debug=False)







