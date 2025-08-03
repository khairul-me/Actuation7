from PyQt5.QtWidgets import *
import cv2
from vimba import *
from  PyQt5.QtGui import *
from  PyQt5.QtCore import *
from  PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from  PyQt5.QtMultimediaWidgets import QVideoWidget
import torch
import json
import os
import threading
from typing import Optional
from time import time, sleep
import numpy as np
import sys
from queue import SimpleQueue
from itertools import count



def checkAndSavePath():
    CheckedDetectedPath=os.path.exists("/media/nvidia/64A5-F009/DetectedImages/")
    CheckedJetsonPath=os.path.exists("/media/nvidia/64A5-F009/jeson/")
    CreatedDetectedFolder=os.getcwd()+"/DetecedImages"
    CreatedJesonFolder=os.getcwd()+"/json"

    if CheckedDetectedPath:
        DetectedPath="/media/nvidia/64A5-F009/DetectedImages/"
    elif os.path.exists(CreatedDetectedFolder):
        DetectedPath = CreatedDetectedFolder+"/"
    else:
        os.mkdir(CreatedDetectedFolder)
        DetectedPath = CreatedDetectedFolder + "/"

    if CheckedJetsonPath:
        json_save_dir = "/media/nvidia/64A5-F009/jeson/"
    elif os.path.exists(CreatedJesonFolder):
        json_save_dir = CreatedJesonFolder + "/"
    else:
        os.mkdir(CreatedJesonFolder)
        json_save_dir = CreatedJesonFolder + "/"

    return DetectedPath,json_save_dir


def  ShowImage():
    image_path = textEdit1.toPlainText()
    if len(image_path)<2 :
        image_path = 'example1.jpg'

    #img_path = 'example1.jpg'  # 设置图片路径
    showImage = QPixmap(image_path).scaled(label_pic.width(), label_pic.height())  # 适应窗口大小
    label_pic.setPixmap(showImage)  # 显


def fault_show_function():
    dialog_fault = QDialog()
    image_path = "example1.jpg"
    pic = QPixmap(image_path)
    label_pic = QLabel("show", dialog_fault)
    label_pic.setPixmap(pic)
    label_pic.setGeometry(10, 10, 500, 500)
    label_pic.setScaledContents(True)
    dialog_fault.exec_()

def  Image_path_dialog():

    global imgPath
    imgPath, imgType = QFileDialog.getOpenFileName(window, "Open image file", "", "*.jpg;;*.png;;All Files(*)")
    showImage = QPixmap(imgPath).scaled(label_pic.width(), label_pic.height())  # 适应窗口大小
    label_pic.setPixmap(showImage)  #
    print(imgType)

def  Video_path_dialog():

    videoPath, _ = QFileDialog.getOpenFileName(window, "open video file",
                                              ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

    playVideoFile(videoPath)

def playVideoFile(videoPath):
        global loop_flag ,moving_flag
        cap=cv2.VideoCapture(videoPath)
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        settingSlider(frames)
        fps = 24
        loop_flag = 0
        if not cap.isOpened():
            print("Cannot open Video File")
            exit()
        while not bClose:
            ret, frame = cap.read()  # read each frame
            if not ret:
                if frame is None:
                    print("The video has end.")
                else:
                    print("Read video error!")
                break
            if moving_flag == 0:
                label_start.setText(int2time(loop_flag))
                positionSlider.setValue(int(loop_flag / 24))  #
            loop_flag += 1
            resize_frame = cv2.resize(frame, (640, 640))
            results = model_detect_one_frame(resize_frame)
            frame = plot_boxes(results, resize_frame)
            QtImg = cvImgtoQtImg(frame)
            label_pic.setPixmap(QPixmap.fromImage(QtImg).scaled(label_pic.size()))
            #label_pic.setPixmap(QPixmap.fromImage(QtImg).scaled(Qt.KeepAspectRatio))
            #label_pic.show()  # reflash
            while stop_flag == 1:  # pause action
                cv2.waitKey(int(1000 / fps))  # sleep
            cv2.waitKey(int(1000 / fps))  # sleep a while
        #  release
        cap.release()




def start_drag():
    moving_flag = 1

def video_stop():
    bClose = True

def stop_action():
    global stop_flag
    if stop_flag == 0:
        stop_flag = 1
        playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPlay))
    else:
        stop_flag = 0
        playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPause))

def  drag_action():
        global moving_flag,loop_flag,cap
        moving_flag = 0
        print('current process is%d，be draged process is%d' % (positionSlider.value(), int(loop_flag / 24)))
        if positionSlider.value() != int(loop_flag / 24):
             print('current process is:' + str(positionSlider.value()))
             loop_flag = positionSlider.value() * 24
             cap.set(cv2.CAP_PROP_POS_FRAMES,loop_flag)


def cvImgtoQtImg(cvImg):  # 定义将opencv图像转PyQt图像的函数
    QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)
    QtImg = QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QImage.Format_RGB32)
    return QtImg


def settingSlider(maxvalue):
        positionSlider.setMaximum(int(maxvalue / 24))
        label_end.setText(int2time(maxvalue))


def int2time( num):
        # 24FPS
        num = int(num / 24)
        minute = int(num / 60)
        second = num - 60 * minute
        if minute < 10:
            str_minute = '0' + str(minute)
        else:
            str_minute = str(minute)
        if second < 10:
            str_second = '0' + str(second)
        else:
            str_second = str(second)
        return str_minute + ":" + str_second




def  Close_dialog():
    reply = QMessageBox.question(window, 'Close', 'Do you want to close current file？', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
    if reply == QMessageBox.Yes:
        label_pic.clear()
    else:
        print("not close")



def show_detect_image():
    #model = torch.hub.load('/home/nvidia/DCW-main/YOLOv5', 'custom', path='bestL.pt', source='local')
    img = cv2.imread(imgPath)
    results = model(img, size=640)
    result_img=results.imgs
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = img.shape[1], img.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(img, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    label_width = label_pic.width()
    label_height = label_pic.height()

    temp_imgSrc = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3,QImage.Format_RGB888)
    pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)

    label_pic.setPixmap(pixmap_imgSrc)


    #print(img)
    #results.show()

def run_realTime():
    os.system("python realTimeWithSave.py")

def sliderH_value():
    label_sliderH.setText("Hue:"+str(slider_H.value())) #做了个强转，不然报错：label框需要str类型值

def sliderS_value():
    label_sliderS.setText("Saturation:" + str(0.1*slider_S.value()))  # 做了个强转，不然报错：label框需要str类型值

def sliderG_value():
    label_sliderG.setText("Gamma:"+str(0.1*slider_G.value())) #做了个强转，不然报错：label框需要str类型值 0.1 in case int

##



def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]






def score_frame(frame):
    """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
    global model
    global device
    i = 0
    while (os.path.exists("js_data/data%s.json" % i)):
        i += 1
    model.to(device)
    frame = [frame]
    results = model(frame)
    # results.save()
    parsed = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    f = open('js_data/data%s.json' % i, 'w')
    json.dump(parsed, f)
    f.close()
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord,i

def getJnumAndInum(DetectedPath,json_save_dir):
    Jnum=1
    Inum=1
    while (os.path.exists(json_save_dir+"data%s.json"% Jnum)):
        Jnum+=1
    while (os.path.exists(DetectedPath+"%s.json"% Inum)):
        Inum+=1
    return Inum,Jnum

def getJnumAndInum1(DetectedPath,json_save_dir):
    Jnum=1
    Inum=1
    maxDetected = 0
    maxJson=0
    for root, dirs, files in os.walk(DetectedPath):
        for file in files:
            ext=os.path.splitext(file)
            if int(ext[0])>= maxDetected:
                maxDetected=int(ext[0])

    #for root, dirs, files in os.walk(json_save_dir):
       # for file in files:
            #ext = os.path.splitext(file)
            #if int(ext[0]) >= maxJson:
               # maxJson = int(ext[0])

    Inum=maxDetected
    Jnum=Inum

    return Inum,Jnum

def model_detect_one_frame(frame):
    global model
    global device

    model.to(device)
    frame = [frame]
    results = model(frame)
    # results.save()
    return results


def save_json(results,json_save_dir,Jnum):
    parsed = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    f = open(json_save_dir+"data%s.json"% Jnum, 'w')
    json.dump(parsed, f)
    f.close()



def only_score_frame(frame):
    """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
    global model
    global device

    model.to(device)
    frame = [frame]
    results = model(frame)
    # results.save()
    #parsed = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    #f = open('js_data/data%s.json' % i, 'w')
    #json.dump(parsed, f)
    #f.close()
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    global model
    classes = model.names
    return classes[int(x)]


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 4)
    return frame

def  frame_handler(cam , frame ):
    opencv_frame=frame.as_opencv_image()
    cv2.imshow("sd",opencv_frame)
    cam.queue_frame(frame)



def Run_realTime():
    #print_preamble()
    cam_id = parse_args()


    with Vimba.get_instance():
        with get_camera(cam_id) as cam:
            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler()
            cam.start_streaming(handler=handler, buffer_count=10)
            handler.shutdown_event.wait()
            cam.TriggerSoftware.run()
            time.sleep(5)
            cam.stop_streaming()

def ModelChange():
    cur_model = combol1.currentText() + ".pt"
    model = torch.hub.load('/home/nvidia/DCW-main/YOLOv5', 'custom', path=cur_model, source='local')
    print("model is added")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def camera_streaming(queue):
    global is_streaming
    is_streaming = True
    print("streaming started")
    message_label.setText("streaming started")
    with Vimba.get_instance() as vimba:
        with vimba.get_all_cameras()[0] as camera:
            while is_streaming:
                #print(is_streaming)
                frame = camera.get_frame()
                #frame = frame.as_opencv_image()
               # im = Image.fromarray(frame)
               # img = ImageTk.PhotoImage(im)
                queue.put(frame) # put the capture image into queue
    print("streaming stopped")
    message_label.setText("streaming stopped")

def show_camera():
        global  frame_num
        global Inum
        global Jnum
        global json_save_dir
        global DetectedPath
        if not queue.empty() and not realTime_check.isChecked():
           frame=queue.get ()
           c_frame=frame.as_opencv_image()
          # c_frame=frame.as_numpy_ndarray()
           #c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BAYER_GR2BGR)
           cv2.putText(c_frame, f'Time: {int(time())}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
           label_width = label_pic.width()
           label_height = label_pic.height()
           c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
           temp_imgSrc = QImage(c_frame, c_frame.shape[1], c_frame.shape[0], c_frame.shape[1] * 3,
                                QImage.Format_RGB888)
           pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
           label_pic.setPixmap(pixmap_imgSrc)
        elif not queue.empty() and  realTime_check.isChecked():
            frame = queue.get()
            c_frame=frame.as_opencv_image()
            start_time=time()
            c_frame=cv2.cvtColor(c_frame, cv2.COLOR_BGR2RGB)
            resize_frame=cv2.resize(c_frame,(640,640))
            results=model_detect_one_frame(resize_frame)

            final_frame=plot_boxes(results,resize_frame)
            end_time=time()
            fps=1/np.round(end_time-start_time,2)
            cv2.putText(final_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            label_width = label_pic.width()
            label_height = label_pic.height()
            temp_imgSrc = QImage(final_frame, final_frame.shape[1], final_frame.shape[0], final_frame.shape[1] * 3,
                                 QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            label_pic.setPixmap(pixmap_imgSrc)
            #print(json_save_dir,results,Jnum)
            print(c_frame.shape)
            if Inum % save_num.value() == 0 and save_check.isChecked():  # some  variable parameters
                img_name = textEdit2.toPlainText() + str(Inum) + ".jpg"
                cv2.imwrite(DetectedPath, final_frame)
                print()
                save_json(results, json_save_dir, Inum)
                print(Inum)
            Inum+=1


def button_open_camera_clicked():
    if timer_camera.isActive() == False:
        threading.Thread(target=camera_streaming, args=(queue,), daemon=True).start()
        #threading.Thread(target=saveImage_Runthread, daemon=True).start()
        timer_camera.start(40)
        button5.setText('Close Camera')
    else:
        global is_streaming
        is_streaming=False
        timer_camera.stop()
        #is_streaming=False
        #cam.stop_streaming()
        label_pic.clear()
        button5.setText("Open Camera")

#def button_close_camera_clicked(): global is_streaming

def saveImage_Runthread():
    global  is_saving
    is_saving= True
    print("Saving Image")
    while is_saving :
            frame_id = c_frame.get_id()
            if frame_id % save_num.value() == 0 and save_check.isChecked():  # some  variable parameters
                img_name = textEdit2.toPlainText() + str(frame_id) + ".jpg"
                cv2.imwrite(img_name, final_frame)
                print(frame_id)
    print("Saving Stop")


        #while save_check.isChecked():
def real_time_button_test(pressed):

    if pressed:
        print("presssssed")
    else:
        print("not pressed")










app=QApplication([])



window=QMainWindow()
window.resize(900,550)
window.move(300,310)
window.setWindowTitle('OpenWeedGUI')
frame_num=1
DetectedPath,json_save_dir=checkAndSavePath()
Inum,Jnum=getJnumAndInum1(DetectedPath,json_save_dir)
print(DetectedPath,json_save_dir)
print(Inum,Jnum)
bClose = False
moving_flag = 0
stop_flag = 0  # pause change it to 1

menuBar = window.menuBar()
#menuBar.resize(300,20)
#fileMenu = QMenu("&File", window)
file=menuBar.addMenu("File")
edit=menuBar.addMenu("Edit")
setting=menuBar.addMenu("Setting")
help=menuBar.addMenu("Help")
open_image= QAction("Open image file", window)
open_video= QAction("Open video file", window)
close= QAction("Close", window)
file.addAction(open_image)
file.addAction(open_video)
file.addAction(close)
open_image.triggered.connect(Image_path_dialog)
open_video.triggered.connect(Video_path_dialog)
close.triggered.connect(Close_dialog)

Image_path_e= QAction("Edit Image path", window)
cam_set= QAction("cam set", window)
edit.addAction(Image_path_e)
setting.addAction(cam_set)



statusBar = window.statusBar()

message_label=QLabel(window)
message_label.resize(300,80)
message_label.move(30,420)
message_label.setStyleSheet("QLabel{background:white;}")
message_label.setText("OpenWeedGui is running")



#qbtn1=QRadioButton(window)
#qbtn1.setText("Image")
#qbtn1.move(30,30)
#qbtn1.setChecked(True)
#qbtn1.clicked.connect(Image_path_dialog)

#qbtn2=QRadioButton(window)
#qbtn2.setText("Video")
#qbtn2.move(130,30)

#qbtn=QRadioButton(window)
#qbtn.setText("RealTime")
#qbtn.move(230,30)

#label = QLabel('YOLOWeeds',window)
#label.resize(300,300)
#label.move(50,100)

"""
image_path = "example1.jpg"
pic = QPixmap(image_path)
label_pic = QLabel("show", window)
label_pic.resize(400,400)
label_pic.setPixmap(pic)
label_pic.move(400,100)
label_pic.setScaledContents (True)
"""
label_pic = QLabel(window)  # 设置图片显示label
label_pic.setText(" Frame")
label_pic.setFixedSize(400, 400)  # 设置图片大小
label_pic.move(400, 60)  # 设置图片位置
label_pic.setStyleSheet("QLabel{background:white;}")  # 设置label底色
###slider###############
#label_pic.setGeometry(10, 10, 1019, 537)



slider_H= QSlider(window)
slider_H.setOrientation(Qt.Horizontal)
slider_H.setMinimum(-40)
slider_H.setMaximum(40)
slider_H.move(200,170)
slider_H.setSingleStep(2)
slider_H.setValue(0)# 步长slider.setValue(18)  # 当前值
#slider_H.setTickPosition.TicksBelow  # 设置刻度的位置，刻度在下方
slider_H.setTickInterval(5)  #
slider_H.setTickPosition(QSlider.TicksBelow)
slider_H.valueChanged.connect(sliderH_value)
label_sliderH=QLabel(window)
label_sliderH.setText("Hue:"+str(slider_H.value()))
label_sliderH.move(200,140)
###
slider_S= QSlider(window)
slider_S.setOrientation(Qt.Horizontal)
slider_S.setMinimum(0)
slider_S.setMaximum(20)
slider_S.move(200,220)
slider_S.setSingleStep(1)
slider_S.setValue(10)# 步长slider.setValue(18)  # 当前值
#slider_H.setTickPosition.TicksBelow  # 设置刻度的位置，刻度在下方
slider_S.setTickInterval(2)  #
slider_S.setTickPosition(QSlider.TicksBelow)
slider_S.valueChanged.connect(sliderS_value)
label_sliderS=QLabel(window)
label_sliderS.setText("Saturation:"+str(0.1*slider_S.value()))
label_sliderS.move(200,200)
###
slider_G= QSlider(window)
slider_G.setOrientation(Qt.Horizontal)
slider_G.setMinimum(4)
slider_G.setMaximum(24)
slider_G.move(200,270)
slider_G.setSingleStep(2)
slider_G.setValue(7)# 步长slider.setValue(18)  # 当前值
#slider_H.setTickPosition.TicksBelow  # 设置刻度的位置，刻度在下方
slider_G.setTickInterval(2)  #
slider_G.setTickPosition(QSlider.TicksBelow)
slider_G.valueChanged.connect(sliderG_value)
label_sliderG=QLabel(window)
label_sliderG.setText("Gamma:"+str(0.1*slider_G.value()))
label_sliderG.move(200,250)


combol1 = QComboBox(window)
combol1.move(250,80)
combol1.addItem("Select Model")
combol1.addItem("bestS")
combol1.addItem("bestX")
combol1.addItem("bestM")
combol1.addItem("bestN")
combol1.addItem("bestL")
combol1.setCurrentIndex(1)

combolY = QComboBox(window)
combolY.move(120,80)
combolY.addItem("Select YOLO")
combolY.addItem("YOLOv5")
combolY.addItem("YOLOv4")
combolY.addItem("YOLOv3")
combolY.setCurrentIndex(1)

label_model=QLabel(window)
label_model.move(10,80)
label_model.setText("Model Selection")

cur_model=combol1.currentText()+".pt"
combol1.currentIndexChanged.connect(ModelChange)
#model = torch.hub.load('D:\\迅雷下载\\YOLOv5\\yolov5s.pt', 'custom', path=cur_model , source='local')
#model = torch.hub.load('D:\\迅雷下载\\YOLOv5\\yolov5s.pt', 'custom', path=cur_model , source='local')
model_loc = r"D:\YOLOv5"
#model = torch.hub.load("D:\\迅雷下载\\yolov5-master", 'custom', path="D:\\YOLOv5\\yolov5s.pt" , source='local')
model = torch.hub.load("/media/agfoodsensinglab/512ssd/WeedGUIProject/DCW-main/YOLOv5", 'custom', path="/media/agfoodsensinglab/512ssd/WeedGUIProject/DCW-main/YOLOv5/bestS.pt" , source='local')
#model = torch.hub.load("D:\\迅雷下载\\yolov5-master", 'custom', path="D:\\迅雷下载\\YOLOv4\\yolov4.pt" , source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

textEdit1 = QPlainTextEdit(window)
textEdit1.setPlaceholderText(" Sources Path")
textEdit1.setPlainText("example1.jpg")
textEdit1.move(30,80)
textEdit1.resize(100,30)
textEdit1.hide()


textEdit2 = QPlainTextEdit(window)
textEdit2.setPlaceholderText(" Save Path")
textEdit2.setPlainText("/media/nvidia/64A5-F009/DetectedImages/")
textEdit2.move(150,80)
textEdit2.resize(100,30)
textEdit2.hide()




button=QPushButton('ShowImage',window)
button.move(30,150)
button.clicked.connect(ShowImage)

button1=QPushButton('DetectImage',window)
button1.move(30,200)
button1.clicked.connect(show_detect_image)

button2=QPushButton('ShowVideo',window)
button2.move(30,250)
button2.clicked.connect(run_realTime)

button3=QPushButton('DetectVideo',window)
button3.move(30,300)
button3.clicked.connect(run_realTime)

#button4=QPushButton('realTime',window)
#button4.move(30,350)
#button4.clicked.connect(real_time_button_test) #run_realTime -----run outdoor

timer_camera = QTimer() #inital some
queue = SimpleQueue() # queue for video frames
timer_camera.timeout.connect(show_camera)

button5=QPushButton('Open Camera',window)
button5.move(30,350)
button5.clicked.connect(button_open_camera_clicked) #run_realTime -----run outdoor

playButton = QPushButton(window)
playButton.setEnabled(False)
playButton.setFixedHeight(24)
playButton.setIconSize(QSize(15, 15))
playButton.setIcon(window.style().standardIcon(QStyle.SP_MediaPlay))
playButton.move(400,480)
playButton.clicked.connect(stop_action)
playButton.hide()
playButton.show()

positionSlider = QSlider(window)
positionSlider.setToolTip("Player Silder")
positionSlider.setMinimum(0)
positionSlider.setMaximum(50)
positionSlider.setSingleStep(1)
positionSlider.setGeometry(530,480,200,10)
positionSlider.setValue(0)
positionSlider.setOrientation(Qt.Horizontal)
positionSlider.sliderMoved.connect(start_drag)
positionSlider.sliderReleased.connect(drag_action)



label_start = QLabel(window)
label_start.setText("00:00")
label_start.move(530,480)
label_end = QLabel(window)
label_end.setText("00:00")
label_end.move(710,480)



#positionSlider.sliderMoved.connect(setPosition)





save_check=QCheckBox(window)
save_check.setText("Save Result")
save_check.setChecked(True)
save_check.move(200,310)

realTime_check=QCheckBox(window)
realTime_check.setText("Real-time Detection")
realTime_check.setChecked(False)
realTime_check.move(550,500)
realTime_check.resize(200,50)


save_num=QSpinBox(window)
save_num.setValue(2)
save_num.move(200,350)
save_num.setSuffix("  frame rate")

window.show()
app. exec_()

