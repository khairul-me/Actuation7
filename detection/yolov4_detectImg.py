import cv2
import sys
import cv2
import torch
sys.path.append("/media/agfoodsensinglab/512ssd/WeedGUIProject/YOLOv4")
from utils4.general import  non_max_suppression4

from utils4.torch_utils import select_device4, load_classifier4, time_synchronized4

from models4.models import Darknet4,load_darknet_weights4

ROOT="/media/agfoodsensinglab/512ssd/WeedGUIProject/YOLOv4"
cfg=ROOT+'/cfg4/yolov4.cfg'
imgsz=(640, 640)
weights="/media/agfoodsensinglab/512ssd/WeedGUIProject/YOLOv4/best_yolov4.pt"
path=ROOT+"/example1.jpg"
img = cv2.imread(path)

device = select_device4("0")
half = device.type != 'cpu'

model = Darknet4(cfg, imgsz).cuda()
# model.load_state_dict(torch.load(weights, map_location={'0':'GPU'})['model'])
try:
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
except:
    load_darknet_weights4(model,weights)

model.to(device).eval()
model.half()

if half:
    model.half()  # to FP16

