import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QLabel, QFileDialog, QMainWindow, QButtonGroup, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, QTimer, QRect
# from  PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from  PyQt5.QtMultimediaWidgets import QVideoWidget
import torch
import json
import os
import threading
from typing import Optional
from time import time, sleep
import datetime
import numpy as np
from queue import SimpleQueue, Queue
from itertools import count
from ultralytics import YOLO
from loguru import logger
import config as cfg
import random
from detection.plot import Plotter
import traceback

TPs = []
FPs = []
FNs = []
GTs = []

class WorkerDetection(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        print('WorkerDetection: start running')
        cfg.global_detector.cur_model = cfg.global_window.comboBox_pt.currentText()
        cfg.global_detector.cur_model_version = cfg.global_window.comboBox_model_name.currentText()
        sleep(0.5)
        print('WorkerDetection: model_init')
        has_exception = False
        try:
            cfg.global_detector.model_init()

            cfg.global_detector.model_is_changed = False
            while True:
                # print(f'Detection loop, camera: {cfg.global_camera_handler.camera_is_opened}, model_is_changed: {cfg.global_detector.model_is_changed}, global_detector: {cfg.global_detector is not None}, detect checked: {cfg.global_window.detect_check.isChecked()}, resize_frame: {cfg.global_detector.resize_frame_for_detect is not None}, results_track: {cfg.global_detector.results_track is not None}, results_grade: {cfg.global_detector.results_grade is not None}')
                sleep(0.001)
                if cfg.global_camera_handler.camera_is_opened is False:
                    break
                if cfg.global_detector.model_is_changed is True:
                    break
                if cfg.global_detector is not None:
                    if cfg.global_window.detect_check.isChecked():
                        if cfg.global_detector.resize_frame_for_detect is not None:
                            start_time = time()
                            try:
                                results = cfg.global_detector.model_detect_one_frame(cfg.global_detector.resize_frame_for_detect)
                                if len(results) == 2:
                                    cfg.global_detector.results_track, cfg.global_detector.results_grade = results
                                else:
                                    print('len(results):', len(results))
                            except Exception as e:
                                print(e)
                                has_exception = True
                                self.finished.emit()
                                break
                            end_time = time()
                            cfg.global_detector.cur_model_fps = np.round(1.0 / (end_time - start_time), 2)
                            # print('fps', cfg.global_detector.cur_model_fps)
                            # print('detected obj num:', len(cfg.global_detector.results_grade.boxes), len(cfg.global_detector.results_track.boxes))
        except Exception as e:
            print(e)
        if not has_exception:
            self.finished.emit()
        cfg.global_detector.model_is_changed = False


class Detector():
    def __init__(self) -> None:
        if not cfg.async_valve_control:
            from sensing.worker_send_signal import WorkerSendSignalValve, WorkerSendSignalSpeed
            self.worker_valve = WorkerSendSignalValve()
        self.loop_idx = 0
        self.plotter = Plotter()
        self.graded_sps = {}

        self.cur_model = None
        self.cur_model_version = None
        self.cur_model_fps = 0
        self.model_is_changed = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_added = False
        self.model_loading = False
        self.json_save_dir = []

        self.resize_frame_for_detect = None
        self.results_track = None
        self.results_grade = None
        
        # self.model_paths = [r'..\model\bestL_track1.pt',
        #             r'..\model\bestL_grade1.pt']
        
    def model_init(self):
        self.model_added = False
        self.model_loading = True
        while not cfg.global_window.model_selection_finished:
            print('wait for model_selection_finished')
            sleep(0.1)

        self.model_track = YOLO(cfg.model_paths['track'])
        self.model_grade = YOLO(cfg.model_paths['detection'])
        # self.class_names = self.model_grade.names
        self.class_names = cfg.sku_names
        print('Class Names: ', self.class_names)

        print("model is added")
        self.model_added = True
        self.model_loading = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model_track.to(self.device)
        self.model_grade.to(self.device)

    def results(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        self.show_statistic(results)
        return frame

    def release_model(self):
        cfg.global_window.statusBar().showMessage("Release the current model.")

    def save_json(self, results, json_save_dir, Jnum):
        f = open(json_save_dir + "data%s.json" % Jnum, 'w')
        json.dump(results, f)
        f.close()

    def idx_to_label(self, idx):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.class_names[int(idx)]

    def main_loop(self):
        self.cur_model_version = cfg.global_window.comboBox_model_name.currentText()
        self.cur_model = cfg.global_window.comboBox_pt.currentText()
        start_time_tot = time()

        msg = ''
        if self.model_loading:
            msg = "model is loading: " + self.cur_model_version
            cfg.global_windowmodel_selection_finished = True
            cfg.global_window.statusBar().showMessage(msg)
        elif self.model_added:
            msg = "model has been loaded: " + self.cur_model_version
            cfg.global_windowmodel_selection_finished = False
            self.model_added = False
            cfg.global_window.statusBar().showMessage(msg)

        try:
            frame = None
            frame_ann = None
            frame_status = None
            if cfg.async_camera_stream:
                if not cfg.global_camera_handler.frame_queue.empty():
                    frame_info = cfg.global_camera_handler.frame_queue.get_nowait()
            else:
                frame_read_status, frame_info = cfg.global_camera_handler.capture()
            if type(frame_info) is dict:
                frame = frame_info['img']
                frame_ann = frame_info['ann']
                frame_status = frame_info['status']
            else:
                frame = frame_info
            if frame is None:
                if frame_status == 'Finished':
                    self.assess_metrics()
                return
            frame_bgr = frame
            # frame = frame.as_opencv_image()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.resize_frame_for_display = cv2.resize(frame_rgb, cfg.display_size)
            final_frame = self.resize_frame_for_display

            results = None

            if cfg.global_window.detect_check.isChecked():
                if frame_ann is not None:
                    self.resize_frame_for_detect = frame_bgr
                else:
                    self.resize_frame_for_detect = cv2.resize(frame_bgr, cfg.detect_size)
                if not cfg.async_detection:
                    self.results_track = None
                    self.results_grade = None
                    start_time = time()
                    self.results_track, self.results_grade = self.model_detect_one_frame(self.resize_frame_for_detect)
                    end_time = time()
                    cfg.global_detector.cur_model_fps = np.round(1.0 / max((end_time - start_time), 0.001), 2)
                try:
                    resize_frame_for_display = self.plotter.plot_valve_location(self.resize_frame_for_display)
                    # if self.results_track is not None and self.results_grade is not None:
                    if self.results_track is not None or self.results_grade is not None:
                        if self.results_track is None:
                            final_frame = self.plotter.plot_boxes_detection_only(self.results_track, self.results_grade, self.resize_frame_for_detect, self.resize_frame_for_display)
                        else:
                            final_frame = self.plotter.plot_boxes_with_track(self.results_track, self.results_grade, self.resize_frame_for_detect, self.resize_frame_for_display)

                        if self.graded_sps != self.plotter.graded_sps:
                            self.graded_sps = self.plotter.graded_sps
                            for key in self.graded_sps:
                                print(key, self.graded_sps[key])
                        results = {'results_track': self.results_track, 'results_grade': self.results_grade}
                    if frame_ann is not None:
                        self.collect_data_for_metrics(self.results_grade, frame_ann)

                except Exception as e:
                    print(traceback.format_exc())
                    print(e)
                if cfg.global_detector.cur_model_fps > 30:
                    fps_display = '30.0+'
                else:
                    fps_display = round(cfg.global_detector.cur_model_fps, 1)
                cv2.putText(final_frame, f'Model FPS: {fps_display}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            label_width = cfg.global_window.label_pic.width()
            label_height = cfg.global_window.label_pic.height()
            temp_imgSrc = QImage(final_frame, final_frame.shape[1], final_frame.shape[0], final_frame.shape[1] * 3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            cfg.global_window.label_pic.setPixmap(pixmap_imgSrc)

            if cfg.global_window.save_check.isChecked():
                cfg.global_image_handler.save_frame(frame_origin=frame_bgr, frame_detected=final_frame, results=results)
            else:
                cfg.global_image_handler.finalize()

            if not cfg.async_valve_control:
                self.worker_valve.run_sync()

            end_time_tot = time()
            if self.loop_idx % 200 == 0:
                cur_tot_fps = np.round(1.0 / (end_time_tot - start_time_tot), 2)
                msg = 'ovarall frame rate:' + str(cur_tot_fps)
                cfg.log_xalg(msg)
                # cfg.global_window.statusBar().showMessage(msg)
            self.loop_idx+=1
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def assess_metrics(self):
        """
        ref to YOLOv8 code:
        ap_per_class(), compute_ap()
        only one class for indoor test 
        """
        global TPs, FPs, FNs, GTs

        TPs_np = np.array(TPs)
        FPs_np = np.array(FPs)
        FNs_np = np.array(FNs)
        GTs_np = np.array(GTs)

        precision_tot = sum(TPs)/(sum(TPs)+sum(FPs))
        recall_tot = sum(TPs)/(sum(TPs)+sum(FNs))

        tpc = TPs_np.cumsum(0)
        gt_num = np.sum(GTs_np)
        gt = GTs_np.cumsum(0)
        recall = tpc / gt

        fpc = FPs_np.cumsum(0)
        precision = tpc / (tpc + fpc)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        return precision_tot, recall_tot, ap

    def collect_data_for_metrics(self, predictions, gts):
        """
        The patterns were sparsely distributed, so no overlap existed
        """
        from metric.util import compute_iou
        global TPs, FPs, FNs, GTs
        TP = 0
        FP = 0
        FN = 0
        GT = len(gts)
        for prediction in predictions:
            has_gt = False
            for gt in gts:
                iou = compute_iou(prediction, gt)
                if iou > 0:
                    TP += 1
                    has_gt = True
                    break
            if has_gt is False:
                    FP += 1

        for gt in gts:
            has_pred = False
            for prediction in predictions:
                iou = compute_iou(prediction, gt)
                if iou > 0:
                    has_pred = True
                    break
            if has_pred is False:
                    FN += 1
        TPs += [TP]
        FPs += [FP]
        FNs += [FN]
        GTs += [GT]

    def color_classify_pattern(self, img):
        config_indoor = False
        if config_indoor:
            lower_H = np.array([75])
            upper_H = np.array([90])

            lower_S = np.array([100])
            upper_S = np.array([120])
        else:
            # green mark
            lower_H = np.array([65])
            upper_H = np.array([100])

            lower_S = np.array([90])
            upper_S = np.array([130])

            # white 
            lower_H = np.array([90])
            upper_H = np.array([130])

            lower_S = np.array([80])
            upper_S = np.array([200])

            lower_V = np.array([100])
            upper_V = np.array([255])

            lower_V1 = np.array([200])
            upper_V1 = np.array([255])

        h = img.shape[0]
        w = img.shape[1]
        area = h*w

        obj_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        (H, S, V) = cv2.split(obj_hsv)

        mask_H = cv2.inRange(H, lower_H, upper_H)
        output_H = cv2.bitwise_and(H, H, mask=mask_H)

        mask_S = cv2.inRange(S, lower_S, upper_S)
        output_S = cv2.bitwise_and(S, S, mask=mask_S)

        mask_V = cv2.inRange(V, lower_V, upper_V)
        output_V = cv2.bitwise_and(V, V, mask=mask_S)

        mask_V1 = cv2.inRange(V, lower_V1, upper_V1)
        output_V1 = cv2.bitwise_and(V, V, mask=mask_S)

        mask_tot = mask_H & mask_S & mask_V 
        mask_tot = mask_tot | mask_V1
        
        output_tot = cv2.bitwise_and(S, S, mask=mask_tot)
        if config_indoor:
            output = output_S
        else:
            # output = output_tot
            output = mask_tot

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        iterations = 4
        output = cv2.dilate(output, kernel, iterations=iterations)
        output = cv2.erode(output, kernel, iterations=iterations)
        #  scikit-image
        from skimage.filters import threshold_yen, threshold_otsu, try_all_threshold
        # thresh = threshold_yen(output)
        ret, output_binary = cv2.threshold(output, 100, 255, cv2.THRESH_BINARY)
        if len(output_binary.shape) ==3:
            output_binary = output_binary[:, :, 1]
        return output, output_binary

    def detect_pattern(self, binary_img):
        """
        binary image
        """
        import cv2
        img_circle = np.stack([binary_img, binary_img, binary_img], axis=-1)
        # binary_img_filtered = cv2.medianBlur(binary_img, 5)

        # circles = cv2.HoughCircles(binary_img, cv2.HOUGH_GRADIENT, 1, 50,
        #                         param1=200, param2=15, minRadius=15, maxRadius=25)
        # circles = np.uint16(np.around(circles))
        # find_circle_list = []
        # for i_circle in range(circles.shape[1]):
        #     (x, y, r) = circles[0][i_circle]
        #     cv2.circle(img_circle, (x, y), r, (0, 0, 255), 2)
        #     cv2.circle(img_circle, (x, y), 2, (0, 255, 0), 3)
        #     find_circle_list.append((x, y, r))

        # use contour? if the circle is filled.
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = 15*15*3.1415
            if area > min_area:
                contours_filtered.append(contour)
        # print(circles.shape)
        cv2.drawContours(img_circle, contours_filtered, -1, (0,255,0), 3)
        return contours_filtered, img_circle

    def detect_pattern_by_morphology(self, frame):
        results_grade = [None]
        output, output_binary = self.color_classify_pattern(frame)
        contours, img_circle = self.detect_pattern(output_binary)
        boxes = []
        #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            box = [x, y, x+w, y+h, 'Target']
            boxes.append(box)
        results_grade = [boxes]
        return results_grade
    
    def model_detect_one_frame(self, frame):
        if cfg.pattern_detection_by_morphology:
            results_grade = self.detect_pattern_by_morphology(frame)
            results_track = [None]
        elif cfg.only_model_detection:
            results_grade = self.model_grade.predict(frame, show=False, save=False, show_labels=False, show_conf=False, conf=0.5, iou=0.4, save_txt=False, verbose=False)
            results_track = [None]
        elif cfg.only_track:
            results_track = self.model_track.track(frame, persist=True, conf=0.5, iou=0.5, save_txt=False, tracker="botsort.yaml", verbose=False)
            results_grade = [None]
        else:
            results_track = self.model_track.track(frame, persist=True, conf=0.5, iou=0.5, save_txt=False, tracker="botsort.yaml", verbose=False)
            results_grade = self.model_grade.predict(frame, show=False, save=False, show_labels=False, show_conf=False, conf=0.5, iou=0.4, save_txt=False, verbose=False)
        results_track = results_track[0]
        results_grade = results_grade[0]
        return results_track, results_grade

