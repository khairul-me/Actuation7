import torch
import numpy as np
import json
import config as cfg


def show_statistic(self, results):
    if self.cur_model_version == "yolov8":
        labels = results[0].boxes.cls
        conf = results[0].boxes.conf
        cord = results[0].boxes.xyxyn
        conf = torch.unsqueeze(conf, dim=1)
        cord = torch.cat((cord, conf), axis=1)
    elif self.cur_model_version == "yolox":
        results = results
        labels = results[:, 6]
        conf = results[:, 4] * results[:, 5]
        conf = torch.unsqueeze(conf, dim=1)
        cord = results[:, 0:4]
        cord = torch.cat((cord, conf), axis=1)
    elif self.cur_model_version == "yolor" or self.cur_model_version == "yolov4":
        results = results
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

    if self.cur_model == "yolov8bestS.pt":
        statistic_dic = {name: 0 for name in self.yolov8_class}
    elif self.cur_model == "yolov8bestS1.pt" or self.cur_model == "best_ckpt1.pth":
        statistic_dic = {name: 0 for name in self.yolov8_class_more}
    else:
        statistic_dic = {name: 0 for name in self.model.names}

    n = len(labels)
    for i in range(n):
        row = cord[i]
        if row[4] >= self.confSpinBox.value():

            statistic_dic[self.class_to_label(labels[i])] += 1
    try:
        self.resultWidget.clear()
        statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
        statistic_dic = [i for i in statistic_dic if i[1] > 0]
        results = [' ' + i[0] + '：' + str(i[1]) for i in statistic_dic]
        self.resultWidget.addItems(results)
    except Exception as e:
        print(repr(e))

def match_track_with_detect(box1, box2s):
    # box1 from track
    # box2s all boxes from grade
    thr = 100
    for idx, box2 in enumerate(box2s):
        box2 = box2[0:4].tolist()
        iou = calculate_iou(box1, box2)
        if iou < thr:
            thr = iou
            final_idx = idx
    if 'final_idx' in locals():
        pass
    else:
        final_idx = 1
        print("final_idx 没有被赋值或绑定")
    return final_idx
    
def calculate_iou(bbox1, bbox2):
    # bbox为(x_min, y_min, x_max, y_max)格式的BBOX
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # 计算BBOX1和BBOX2的交集区域
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # 计算BBOX1和BBOX2的并集区域
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算IoU值
    iou = intersection_area / union_area

    return iou

def combine_measurements(measurements, confidences):
    # Initialize the total confidence and the weighted sum for each grade
    
    total_confidence = 0.0
    weighted_sum = {grade: 0.0 for grade in cfg.readable_name_to_sku_name_dict}

    # Iterate over each measurement and confidence level
    for measurement, confidence in zip(measurements, confidences):
        # If grade3 is detected with high confidence, return grade3 immediately
        if measurement == cfg.readable_name_to_sku_name_dict["Severe defects"] and confidence >= 0.9:
            return cfg.readable_name_to_sku_name_dict["Severe defects"]

        # Update the weighted sum for the corresponding grade
        weighted_sum[measurement] += confidence
        total_confidence += confidence

    # Calculate the weighted average for each grade
    weighted_average = {grade: weighted_sum[grade] / total_confidence for grade in cfg.readable_name_to_sku_name_dict}

    # Determine the final result based on the highest weighted average
    final_result = max(weighted_average, key=weighted_average.get)
    return final_result

def save_data(object_records):
    # Save data to a file
    with open("data.json", "w") as file:
        json.dump(object_records, file)
        print("success")
