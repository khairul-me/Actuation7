import json
import math
import cv2
import random
import numpy as np
from os.path import join
import os


def list_dir(path, list_name, extension):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                list_name.append(file_path)
    try:
        list_name = sorted(list_name, key=lambda k: int(
            os.path.split(k)[1].split(extension)[0].split('_')[-1]))
    except Exception as e:
        print(e)
    return list_name


def motMetricsEnhancedCalculator_demo(gtSource, tSource):
    """
    <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
    """
    # import required packages
    import motmetrics as mm
    import numpy as np

    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:],
                                    max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(), C)

        mh = mm.metrics.create()

        summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr',
                                           'recall', 'precision', 'num_objects',
                                           'mostly_tracked', 'partially_tracked',
                                           'mostly_lost', 'num_false_positives',
                                           'num_misses', 'num_switches',
                                           'num_fragmentations', 'mota', 'motp'
                                           ],
                             name='acc')

        strsummary = mm.io.render_summary(
            summary,
            # formatters={'mota' : '{:.2%}'.format},
            namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                     'precision': 'Prcn', 'num_objects': 'GT', \
                     'mostly_tracked': 'MT', 'partially_tracked': 'PT', \
                     'mostly_lost': 'ML', 'num_false_positives': 'FP', \
                     'num_misses': 'FN', 'num_switches': 'IDsw', \
                     'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP',  \
                     }
        )
        print(strsummary)


def motMetricsEnhancedCalculator(im_gt_dict, im_detect_dict):
    """
    <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
    """
    # import required packages
    import motmetrics as mm

    im_names = list(im_gt_dict.keys())
    im_names1 = list(im_detect_dict.keys())
    assert len(im_names) == len(im_names1)

    frame_num = len(im_gt_dict)

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # for origin im size
    roi_x0 = 1533
    roi_x1 = 387
    # Max frame number maybe different for gt and t files
    for i_frame in range(frame_num):
        im_name = im_names[i_frame]
        boxes_gt = im_gt_dict[im_name]
        im_name1 = im_names1[i_frame]
        boxes_detect = im_detect_dict[im_name1]
        assert im_name.split('.')[0] == im_name1.split('.')[0]
        gt_dets = []
        for id in boxes_gt:
            xmin, ymin, xmax, ymax = boxes_gt[id]
            if xmax > roi_x0 or xmin < roi_x1:
                continue
            w = xmax-xmin
            h = ymax-ymin
            gt_dets.append([int(id), xmin, ymin, w, h])
        t_dets = []
        for id in boxes_detect:
            xmin, ymin, xmax, ymax = boxes_detect[id]
            if xmax > roi_x0 or xmin < roi_x1:
                continue
            w = xmax-xmin
            h = ymax-ymin
            t_dets.append([int(id), xmin, ymin, w, h])
        if len(gt_dets) == 0 and len(t_dets) == 0:
            continue

        if len(gt_dets) == 0:
            gt_dets.append([int(-1), 0, 0, 1, 1])
        if len(t_dets) == 0:
            t_dets.append([int(-1), 0, 0, 1, 1])

        gt_dets = np.array(gt_dets)
        t_dets = np.array(t_dets).astype(int)
        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:],
                                    max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr',
                                       'recall', 'precision', 'num_objects',
                                       'mostly_tracked', 'partially_tracked',
                                       'mostly_lost', 'num_false_positives',
                                       'num_misses', 'num_switches',
                                       'num_fragmentations', 'mota', 'motp'
                                       ],
                         name='acc')

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                 'precision': 'Prcn', 'num_objects': 'GT', \
                 'mostly_tracked': 'MT', 'partially_tracked': 'PT', \
                 'mostly_lost': 'ML', 'num_false_positives': 'FP', \
                 'num_misses': 'FN', 'num_switches': 'IDsw', \
                 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP',  \
                 }
    )
    print(strsummary)


def read_xml(in_path):
    '''读取并解析xml文件'''
    from xml.etree.ElementTree import ElementTree
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def if_match(node, kv_map):
    '''判断某个节点是否包含所有传入参数属性
      node: 节点
      kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


def get_node_by_keyvalue(nodelist, kv_map):
    '''根据属性及属性值定位符合的节点，返回节点
      nodelist: 节点列表
      kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


def find_nodes(tree, path):
    '''查找某个路径匹配的所有节点
      tree: xml树
      path: 节点路径'''
    return tree.findall(path)


def load_gt():
    label_files = list_dir(
        r'C:\Users\AFSALab\OneDrive - Michigan State University\Project\SweetPotatoSorting\video_data\gt\1', [], '.xml')
    im_gt_dict = {}
    for label_file in label_files:
        im_name = os.path.basename(label_file)
        im_gt_dict[im_name] = {}
        tree = read_xml(label_file)
        filename_nodes = get_node_by_keyvalue(find_nodes(tree, "filename"), {})
        object_nodes = get_node_by_keyvalue(find_nodes(tree, "object"), {})

        name_nodes = get_node_by_keyvalue(find_nodes(tree, "object/name"), {})
        xmin_nodes = get_node_by_keyvalue(
            find_nodes(tree, "object/bndbox/xmin"), {})
        ymin_nodes = get_node_by_keyvalue(
            find_nodes(tree, "object/bndbox/ymin"), {})
        xmax_nodes = get_node_by_keyvalue(
            find_nodes(tree, "object/bndbox/xmax"), {})
        ymax_nodes = get_node_by_keyvalue(
            find_nodes(tree, "object/bndbox/ymax"), {})
        for index, node in enumerate(object_nodes):
            xmin = int(xmin_nodes[index].text)
            ymin = int(ymin_nodes[index].text)
            xmax = int(xmax_nodes[index].text)
            ymax = int(ymax_nodes[index].text)
            id = name_nodes[index].text
            im_gt_dict[im_name][id] = [xmin, ymin, xmax, ymax]
    return im_gt_dict


def load_detection():
    with open(r'C:\Users\AFSALab\OneDrive - Michigan State University\Project\SweetPotatoSorting\video_data\detect\1\result_dict.json', 'r') as f:
        im_detect_dict = json.load(f)
    return im_detect_dict


def main():
    im_gt_dict = load_gt()
    im_detect_dict = load_detection()
    motMetricsEnhancedCalculator(im_gt_dict, im_detect_dict)


if __name__ == '__main__':
    main()
