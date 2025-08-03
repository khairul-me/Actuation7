# First import the library
from camera_abstract import CameraBase
import cv2
import config as cfg
from metric.util import get_ann_info_from_xml

use_video = False
use_img = not use_video
with_ann = True

class CameraOpenCV(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        if use_video:
            video_path = r"C:\Users\15172\OneDrive - Michigan State University\Project\WeedRobot\Test Indoor\rep3_25patches\video_origin_13_36_11.008589.avi"
            self.cap = cv2.VideoCapture(video_path)
            # self.cap = cv2.VideoCapture(0)
        elif use_img:
            im_dir = r'C:\Users\15172\OneDrive - Michigan State University\Project\WeedRobot\TestIndoor\test_set'
            self.im_paths = cfg.list_dir(im_dir, [], '.jpg')
            if with_ann:
                self.ann_paths = cfg.list_dir(im_dir, [], '.xml')
            self.im_idx = 0

    def capture(self):
        success = True
        img = None
        ann = None
        try:
            if use_video:
                success, img = self.cap.read()
            elif use_img:
                if self.im_idx == -1:
                    success = False
                elif self.im_idx >= len(self.im_paths):
                    self.im_idx = -1
                else:
                    self.cur_img_path = self.im_paths[self.im_idx]
                    img = cv2.imread(self.cur_img_path)
                    if with_ann:
                        self.cur_ann_path = self.ann_paths[self.im_idx]
                        ann = get_ann_info_from_xml(self.cur_ann_path)
        except Exception as e:
            print(e)
            success = False
        self.im_idx += 1
        if success:
            img_info_dict = {'img': img, 'ann': ann, 'status': 'OK'}
        else:
            img_info_dict = {'img': img, 'ann': ann, 'status': 'Finished'}
        # depth = frames.get_depth_frame()
        # if not depth: 
        #     continue
        return success, img_info_dict

    def end(self):
        self.cap.release()

def demo():
    pass


def main():
    demo

if __name__ =='__main__':
    main()

# import numpy as np
# depth = frames.get_depth_frame()
# depth_data = depth.as_frame().get_data()
# np_image = np.asanyarray(depth_data)