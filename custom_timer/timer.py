import sys
import cv2
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
# from  PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from  PyQt5.QtMultimediaWidgets import QVideoWidget
import torch
import json
import os
import threading
from typing import Optional
from time import time, sleep
import numpy as np
from queue import SimpleQueue, Queue
import config as cfg
from datetime import datetime


class Timer():
    def __init__(self) -> None:
        self.timer_camera = QTimer()  # inital some
        self.timer_video = QTimer()
        self.timer_img = QTimer()  # new
        self.timer_img.timeout.connect(cfg.global_image_handler.show_img)  # new
        self.timer_video.timeout.connect(cfg.global_video_handler.show_video)
        self.timer_camera.timeout.connect(cfg.global_camera_handler.show_camera)
        self.last_time = -1

    def get_cur_time(self, is_init=False):
        if is_init:
            current_time = datetime.now()
            a_timedelta = current_time - datetime(1900, 1, 1)
            current_time = int(a_timedelta.total_seconds())
        else:
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S_%f")
        return current_time

    def calculate_time_gap(self, cur_time, last_time):
        format = "%H_%M_%S_%f"
        current_time = datetime.strptime(cur_time, format)
        last_time = datetime.strptime(last_time, format)
        gap = current_time - last_time
        gap = gap.total_seconds()
        return gap
