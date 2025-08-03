import sys
import cv2
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

# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess, vis

import ctypes
import random
import config as cfg
from detection.util import match_track_with_detect, combine_measurements, save_data

def GetFinalGrade(Final_list):

    id, m_len, mid_len, m_wid, mid_wid, f_grad=Final_list
    if mid_len<9 and mid_len>=6 and mid_wid <3.5 and (f_grad=="Normal" or f_grad=="Minor defects"):
        F_SampleGrade="Premium"
    elif mid_len<6 and mid_len>=3 and mid_wid <3.5 and (f_grad=="Normal" or f_grad=="Minor defects"):
        F_SampleGrade = "Good"
    elif mid_wid <3.5 and f_grad=="Severe defects":
        F_SampleGrade = "Fair"
    else:
        F_SampleGrade = "Cull"
    return F_SampleGrade

class Plotter():
    def __init__(self) -> None:
        self.object_records = {}
        self.frame_counter = {}
        self.skip_frames=5
        self.cnt_frame = 0
        self.graded_sps = {}

    def plot_boxes(self, model, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        self = model
        if self.cur_model_version == "yolov8":
            labels = results[0].boxes.cls
            conf = results[0].boxes.conf
            cord = results[0].boxes.xyxyn
            conf = torch.unsqueeze(conf, dim=1)
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolox":
            if self.cur_model == "best_ckpt1.pth":
                size_ratio = 800.0/640
            else:
                size_ratio = 1.0
            results = results[0]
            if results is None:
                return frame
            labels = results[:, 6]
            conf = results[:, 4]*results[:, 5]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]*size_ratio
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolor" or self.cur_model_version == "yolov4":
            results = results[0]
            labels = results[:, 5]
            conf = results[:, 4]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model == "bestS.engine":
            cord = np.concatenate((results[0], results[1].reshape(-1, 1)), axis=1)
            labels = results[2]
        else:
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= self.confSpinBox.value():
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                if self.cur_model_version == "yolox" or self.cur_model == "bestS.engine":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                elif self.cur_model_version == "yolor":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                bgr = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 3)
        self.show_statistic(results)
        return frame

    def plot_oriboxes(self, results, c_frame):
        if self.cur_model_version == "yolov8":
            labels = results[0].boxes.cls
            conf = results[0].boxes.conf
            cord = results[0].boxes.xyxyn
            conf = torch.unsqueeze(conf, dim=1)
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolox":
            if self.cur_model == "best_ckpt1.pth":
                size_ratio = 800.0 / 640
            else:
                size_ratio = 1.0
            results = results[0]
            if results is None:
                return c_frame
            labels = results[:, 6]
            conf = results[:, 4]*results[:, 5]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]*size_ratio
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model_version == "yolor" or self.cur_model_version == "yolov4":
            results = results[0]
            labels = results[:, 5]
            conf = results[:, 4]
            conf = torch.unsqueeze(conf, dim=1)
            cord = results[:, 0:4]
            cord = torch.cat((cord, conf), axis=1)
        elif self.cur_model == "bestS.engine":

            cord = np.concatenate((results[0], results[1].reshape(-1, 1)), axis=1)
            labels = results[2]
        else:
            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        x_shape, y_shape = c_frame.shape[1], c_frame.shape[0]
        scale_x = x_shape / 717
        scale_y = y_shape / 669
        for i in range(n):
            row = cord[i]
            if row[4] >= self.confSpinBox.value():

                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                if self.cur_model_version == "yolox" or self.cur_model == "bestS,engine":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    x1, y1, x2, y2 = x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y
                elif self.cur_model_version == "yolor":
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                bgr = (255, 0, 0)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(c_frame, (x1, y1), (x2, y2), bgr, 3)
                cv2.putText(c_frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 3)
        return c_frame
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=2):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        random.seed(3)
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

            if cfg.plot_improve_label_visibility:
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            else:
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
            
    def overlay(self, image, mask, color, alpha, resize=None):
        """Combines image and its segmentation mask into a single image.

        Params:
            image: Training image. np.ndarray,
            mask: Segmentation mask. np.ndarray,
            color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
            alpha: Segmentation mask's transparency. float = 0.5,
            resize: If provided, both image and its mask are resized before blending them together.
            tuple[int, int] = (1024, 1024))

        Returns:
            image_combined: The combined image. np.ndarray

        """
        # color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
        return image_combined

    # def plot_boxes_with_track_sweetpotato(self, results_track, results_grade, resize_frame_for_detect, resize_frame_for_display):
    def plot_boxes_with_track(self, results_track, results_grade, resize_frame_for_detect, resize_frame_for_display):
        self.cnt_frame += 1
        #offset = 0.0494  # in/px
        offset = 0.0332
        # line_left_x = 129
        line_left_x = 50
        # line_right_x = 640 - 129
        line_right_x = cfg.detect_size[1]-50

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in cfg.global_detector.class_names]
        colors = [[0, 191, 255], [205, 255, 121], [180, 105, 255]]
        h_detect, w_detect = resize_frame_for_detect.shape[:2]
        h_display, w_display = resize_frame_for_display.shape[:2]
        h_ratio = h_display / h_detect
        w_ratio = w_display / w_detect
        img = resize_frame_for_display
        if results_track.masks is not None and results_track.boxes.id is not None:
            masks = results_track.masks.data.cpu()
            ids = results_track.boxes.id.cpu().numpy().astype(int)
            # append_to_xlsx(ids, xlsx_file_path)
            for seg, box, id in zip(masks.data.cpu().numpy(), results_track.boxes, ids):
                seg = cv2.resize(seg, (w_detect, h_detect))
                seg1 = seg.astype(np.uint8)
                res = cv2.findContours(seg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # _, contours, _ = res
                contours, _ = res
                contour = contours[0]
                bboxes = results_grade.boxes.data.cpu().numpy()
                if len(contour) >= 20 and len(bboxes) > 0:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    long_axis = max(axes)
                    short_axis = min(axes)
                    # Compute endpoints of long axis
                    long_axis_angle_rad = np.deg2rad(angle)
                    long_axis_length = long_axis / 2
                    sin_angle = np.sin(long_axis_angle_rad)
                    cos_angle = np.cos(long_axis_angle_rad)
                    pt1 = (int(center[0] - long_axis_length * sin_angle), int(center[1] + long_axis_length * cos_angle))
                    pt2 = (int(center[0] + long_axis_length * sin_angle), int(center[1] - long_axis_length * cos_angle))

                    # Compute endpoints of short axis
                    short_axis_angle_rad = np.deg2rad(angle + 90)
                    short_axis_length = short_axis / 2
                    sin_angle = np.sin(short_axis_angle_rad)
                    cos_angle = np.cos(short_axis_angle_rad)
                    pt3 = (int(center[0] - short_axis_length * sin_angle), int(center[1] + short_axis_length * cos_angle))
                    pt4 = (int(center[0] + short_axis_length * sin_angle), int(center[1] - short_axis_length * cos_angle))

                    # Draw the long and short axes
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    cv2.line(img, pt3, pt4, (0, 255, 0), 2)

                    # Display the lengths of the long and short axes
                    length = long_axis_length*2*offset
                    width = short_axis_length*2*offset

                    text = f"L: {length:.2f}, W: {width:.2f}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_position = (int(center[0] - text_size[0] / 2), int(center[1] + long_axis_length + 20))

                    xmin = int(box.data[0][0])
                    ymin = int(box.data[0][1])
                    xmax = int(box.data[0][2])
                    ymax = int(box.data[0][3])

                    box_track = [xmin, ymin, xmax, ymax]
                    # Bs=results_grade.boxes.data.cpu().numpy()
                    boxes_grade = bboxes

                    final_idx = match_track_with_detect(box_track, boxes_grade)
                    sku = boxes_grade[final_idx][5]

                    img = self.overlay(img, seg, colors[int(sku)], 0.4)

                    if xmin >= line_left_x:
                        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.putText(img, f"Id {id}", (xmax-10, ymax-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, )

                    self.plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(sku)], f'{cfg.global_detector.class_names[int(sku)]} {float(boxes_grade[final_idx][4]):.3}')

                    cv2.line(img, (line_right_x, 0), (line_right_x, 384), (0, 100, 255), 2)
                    cv2.line(img, (line_left_x, 0), (line_left_x, 384), (0, 100, 255), 2)

                    cv2.putText(img, "Record", (line_right_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img, "Grade", (line_left_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # if xmax >= line_position and xmin <= line_position:
                    # if xmax >= line_position and xmin <= line_position:
                    if xmin <= line_right_x:
                        if xmax >= line_right_x:
                            cv2.putText(img, f"Start record {id}", (xmin-10, int((ymax+ymin)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (100, 255, 100), 2)
                        # Record object properties
                        if id not in self.object_records:
                            self.object_records[id] = []
                            self.frame_counter[id] = 0
                        if self.frame_counter[id] % self.skip_frames == 0:
                            self.object_records[id].append({
                                # 'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                                'frame': self.cnt_frame,
                                'class': cfg.global_detector.class_names[int(sku)],
                                "confidence": boxes_grade[final_idx][4],
                                'length': length,
                                'width': width
                            })

                    if xmax-20 >= line_left_x and xmin <= line_left_x:
                        cv2.putText(img, f"Start grade {id}", (xmin - 10, int((ymax + ymin) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
                        lengths = [record.get('length') for record in self.object_records[id]]
                        if lengths is None:
                            print("none")
                        widths = [record['width'] for record in self.object_records[id]]
                        grades = [record['class'] for record in self.object_records[id]]
                        t_confidences = [record['confidence'] for record in self.object_records[id]]
                        m_len = round(sum(lengths)/len(lengths),3)
                        m_wid = round(sum(widths)/len(widths), 3)
                        mid_len = round(np.median(np.array(lengths)),3)
                        mid_wid = round(np.median(np.array(widths)),3)
                        confidences = [tensor.item() for tensor in t_confidences]
                        f_grade_surface = combine_measurements(grades, confidences)
                        # print((id, m_len, mid_len, m_wid, mid_wid, f_grade_surface))

                        f_list = (id, m_len, mid_len, m_wid, mid_wid, f_grade_surface)
                        f_grade_sample= GetFinalGrade(f_list)

                        text = f"F_L: {m_len:.2f}, F_W: {m_wid:.2f}, {f_grade_sample}"

                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_position = (int(center[0] - text_size[0] / 2), int(center[1] + long_axis_length + 20))
                        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        key = cv2.waitKey(10)
                        if key & 0xFF == ord('s'):
                            save_data(self.object_records)
                        if f_grade_sample != cfg.sku_names[0]:
                            if id not in self.graded_sps:
                                #memory increasing
                                self.graded_sps[id] = {'id': id, 'm_len':m_len, 'mid_len':mid_len, 'm_wid':m_wid, 'mid_wid':mid_wid, 'f_grade_surface': f_grade_surface, 'f_grade_sample': f_grade_sample}
                            # del self.object_records[id]
        return img

    def plot_boxes_detection_only(self, results_track, results_grade, resize_frame_for_detect, resize_frame_for_display):
        self.cnt_frame += 1
        offset = 0.0332
        line_left_x = 0
        line_right_x = 10000
        
        h_detect, w_detect = resize_frame_for_detect.shape[:2]
        h_display, w_display = resize_frame_for_display.shape[:2]
        h_ratio = h_display / h_detect
        w_ratio = w_display / w_detect
        img = resize_frame_for_display
        if type(results_grade) is list:
            bboxes = results_grade
        else:
            bboxes = results_grade.boxes.data.cpu().numpy()
        for box in bboxes:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            if len(box) >= 6:
                conf = int(round(box[4], 2)*100)
                sku = int(box[5])
                color = cfg.sku_colors[int(sku)]
                label = f'{conf}% {cfg.global_detector.class_names[int(sku)]}'
            else:
                # pattern_detection_by_morphology
                conf = 100
                color = (120, 194, 126)
                label = f'{conf}% Target'
            self.plot_one_box([xmin, ymin, xmax, ymax], img, color, label)
        return img

    def plot_valve_location(self, img):
        h_im, w_im = cfg.detect_size[:2]
        range_per_valve = h_im/cfg.valve_num
        valve_locations = np.arange(range_per_valve//2, (h_im) + range_per_valve//2, range_per_valve)
        # valve_locations += range_per_valve//2
        for i_valve, valve_location in enumerate(valve_locations):
            valve_location = round(valve_location)
            cv2.line(img, (0, valve_location), (50, valve_location), (255, 128, 0), 3)
            cv2.line(img, (50, valve_location), (100, valve_location), (0, 255, 0), 3)
            cv2.putText(img, str(i_valve+1), (5, valve_location-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)
        return img