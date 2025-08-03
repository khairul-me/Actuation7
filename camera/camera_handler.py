
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  

# from camera_realsense import RealSense
from queue import SimpleQueue, Queue
from os.path import join
import threading
from time import time, sleep
import config as cfg


class CameraHandler():
    def __init__(self) -> None:
        self.camera_is_opened = False
        self.frame_queue = Queue(maxsize=cfg.frame_buffer_count)  # queue for video frames


    def capture(self):
        res, color = self.camera.capture()
        return res, color

    def end(self):
        self.camera.end()

    def open_camera(self):
        try:
            if cfg.camera_vendor == 'vimba':
                from camera_vimba import CameraVimba
                self.camera = CameraVimba()
            elif cfg.camera_vendor == 'realsense':
                from camera_realsense import RealSense
                self.camera = RealSense()
            elif cfg.camera_vendor == 'opencv':
                from camera_opencv import CameraOpenCV
                self.camera = CameraOpenCV()          
            self.camera.check_camera()
            self.camera.setup_camera()
        except Exception as e:
            print(e)

        if cfg.global_timer.timer_camera.isActive() is False:
            if cfg.async_camera_stream:
                threading.Thread(target=self.camera.camera_streaming, args=(self.frame_queue,), daemon=True).start()
            cfg.global_timer.timer_camera.start(25)
            cfg.global_window.OpenCamButton.setText('Close Camera')
            sleep(0.1)
            cfg.global_window.launch_detect_thread()
            self.camera_is_opened = True
        else:
            self.camera.is_streaming = False
            cfg.global_timer.timer_camera.stop()
            cfg.global_window.label_pic.clear()
            cfg.global_window.OpenCamButton.setText("Open Camera")
            self.camera_is_opened = False

    def show_camera(self):
        pass

    def set_exposure(self):
        if cfg.global_window.Exposure_offBox.isChecked():
            print("exposure off")
            cfg.global_window.ExpState="Off"
        elif cfg.global_window.Exposure_onceBox.isChecked():
            print("exposure once")
            cfg.global_window.ExpState="Once"
        elif cfg.global_window.Exposure_continueBox.isChecked():
            print("exposure continue")
            cfg.global_window.ExpState="Continuous"
        else:
            pass


def main():
    import cv2
    cam = CameraHandler()
    img = None
    # save_dir = r'C:\Users\E-ITX\Desktop\BoyangDeng\image'
    save_dir = r'/home/weeding/WeedRobot/test/'
    cnt = 0
    while True:
        k = cv2.waitKey(50)
        if k == ord('q'):
            break
        elif k == ord('s'):
            if img is not None:
                save_path = join(save_dir, str(cnt)+'.jpg')
                cnt += 1
                cv2.imwrite(save_path, img)
                print(save_path)
        else:
            success, img = cam.capture()
        if success:
            cv2.namedWindow('x', 0)
            cv2.imshow('x', img)
    cam.end()


if __name__ == '__main__':
    main()
