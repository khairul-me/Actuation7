# from vimba import *
# from vimba import Vimba, intersect_pixel_formats, OPENCV_PIXEL_FORMATS, COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
import vmbpy 
from queue import Queue
import numpy as np
from time import time, sleep

from vmbpy import *
from vmbpy import intersect_pixel_formats, OPENCV_PIXEL_FORMATS, COLOR_PIXEL_FORMATS, MONO_PIXEL_FORMATS
Vimba=vmbpy.VmbSystem

opencv_display_format = PixelFormat.Bgr8

# First import the library
from camera_abstract import CameraBase
import cv2
from functools import partial
import datetime
try:
    import config as cfg
except Exception as e:
    cfg = None
    print(e)


class Handler:
    def __init__(self):
        self.display_queue = Queue(10)

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # print('{} acquired {}'.format(cam, frame), flush=True)

            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(opencv_display_format)

            self.display_queue.put(display.as_opencv_image(), True)

        cam.queue_frame(frame)

handler = Handler()

def frame_handler(queue, cam, frame):
    img = frame.as_numpy_ndarray()
    frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_CLOCKWISE)
    frame_np = frame_np[:, :, ::-1]
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    try:
        # Try to put the frame in the queue...
        queue.put_nowait(frame_np)
    except queue.Full:
        # If that fials (queue is full), just drop the frame
        # NB: You may want to handle this better...
        print('Dropped Frame')
    cam.queue_frame(frame)

class CameraVimba(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        # cap = cv2.VideoCapture('test.mp4')
        # cap = cv2.VideoCapture(r"C:\Users\AFSALab\OneDrive - Michigan State University\Project\SweetPotatoSorting\video\13.avi")
        # self.cap = cv2.VideoCapture(0)
        self.cam = None
        self.window = None

    def capture(self):
        # success, img = self.cap.read()
        success=True
        # with Vimba.get_instance() as vimba:
        #     with vimba.get_all_cameras()[0] as camera:
        # self.cam = camera
        frame_np = handler.get_image()
        # frame = camera.get_frame()
        # frame_np = frame.as_numpy_ndarray()
        # frame_np = frame_np[:, :, ::-1]

        # no need, directly set in initilization
        # frame_np = frame_np[:, 300:, :]
        # print(frame_np.shape)
        frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_CLOCKWISE)
        
        # depth = frames.get_depth_frame()
        # if not depth:
        #     continue
        return success, frame_np

    def end(self):
        self.cap.release()

    def config(self):
        with Vimba.get_instance() as vimba:
            with vimba.get_all_cameras()[0] as camera:
                camera.ExposureAuto.set(self.ExpState)
                # width_feature = camera.getFeatureByName('Width')
                # print(width_feature)

    def camera_streaming(self, queue):
        self.is_streaming = True
        print("streaming started")
        cfg.global_window.statusBar().showMessage("streaming started")
        idx = 0
        with Vimba.get_instance() as vimba:
            with vimba.get_all_cameras()[0] as cam:
                cam.start_streaming(handler=handler, buffer_count=cfg.frame_buffer_count)
                while self.is_streaming:
                    # f=camera.AcquireSingleImage()
                    frame = None
                    frame_np = None
                    try:
                        # camera.start_streaming(handler=partial(frame_handler, queue) , buffer_count=cfg.frame_buffer_count)
                        # frame = camera.get_frame()
                        # frame_np = frame.as_numpy_ndarray()
                        # # frame_np = frame_np[:, :, ::-1]
                        # frame_np = cv2.rotate(frame_np, cv2.ROTATE_90_CLOCKWISE)

                        idx += 1
                        start_time_tot = time()
                        res, frame_np = self.capture()
                        end_time_tot = time()
                        if idx % 100 == 0:
                            cur_tot_fps = np.round(1.0 / (end_time_tot - start_time_tot), 2)
                            # print('camera fps:', cur_tot_fps)
                            msg = 'camera frame rate:' + str(cur_tot_fps) + ' '+ str(frame_np.shape)
                            cfg.log_xalg(msg)
                            # cfg.global_window.statusBar().showMessage(msg)
                    except Exception as e:
                        print(e)
                    # frame = frame.as_opencv_image()
                    # im = Image.fromarray(frame)
                    # img = ImageTk.PhotoImage(im)

                    if queue.full():
                        with queue.mutex:
                            queue.queue.clear()
                    if frame_np is not None:
                        # queue.put(frame_np)  # put the capture image into queue
                        queue.put_nowait(frame_np)  # put the capture image into queue
        print("streaming stopped")
        cfg.global_window.statusBar().showMessage("streaming stopped")

    def show_camera(self):
        pass

    def set_nearest_value(self, cam: Camera, feat_name: str, feat_value: int):
        # Helper function that tries to set a given value. If setting of the initial value failed
        # it calculates the nearest valid value and sets the result. This function is intended to
        # be used with Height and Width Features because not all Cameras allow the same values
        # for height and width.
        feat = cam.get_feature_by_name(feat_name)

        try:
            feat.set(feat_value)

        except VmbFeatureError:
            min_, max_ = feat.get_range()
            inc = feat.get_increment()

            if feat_value <= min_:
                val = min_

            elif feat_value >= max_:
                val = max_

            else:
                val = (((feat_value - min_) // inc) * inc) + min_

            feat.set(val)

            msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
                'Using nearest valid value \'{}\'. Note that, this causes resizing '
                'during processing, reducing the frame rate.')
            Log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))
            
    def setup_camera(self, cam=None):
        window = self.window
        if cam is not None:
            # cam.set_pixel_format(PixelFormat.BayerRG8)
            # cam.ExposureTimeAbs.set(10000)
            # cam.BalanceWhiteAuto.set('Off')
            # cam.Gain.set(0)
            # cam.AcquisitionMode.set('Continuous')
            # cam.GainAuto.set('Off')
            # # NB: Following adjusted for my Manta G-033C
            # cam.Height.set(492)
            # cam.Width.set(656)
            # Enable auto exposure time setting if camera supports it
            try:
                """
                    These settings cannot be supported by yhis vimba camera:
                    1800 U-507c-01J5H
                    ExposureAutoOutliers
                    Range: [0 to 1000]
                    Default: 0
                    Unit: 0.01% i.e. 1000 = 10%
                    ExposureAutoOutliers is the percentage of pixels on the upper bound of the
                    image brightness distribution graph that are ignored by the ExposureAuto
                    algorithm.
                    
                    ExposureAutoAdjustTol
                    Range: [0 to 50]
                    Default: 5

                    ExposureAutoAlg
                    Mean / FitRange

                    ExposureAutoMax
                    Default: 500000

                    ExposureAutoMin
                    Default: Camera dependent

                    ExposureAutoRate
                    Range: [1 to 100]
                    Default: 100

                    ExposureAutoTarget
                    Range: [0 to 100]
                    Default: 50

                    ExposureMode
                    Auto / Manual / PieceWiseLinearHDR
                    ExposureValue / ExposureValuePWL1 / ExposureValuePWL2
                    ThresholdPWL1 / ThresholdPWL2
                    ThresholdPWL2 < ThresholdPWL1
                    Range: [0 to 63]
                    Default: 63
                    0 = no light capacity, 63 = full pixel light capacity.
                """

                # cam.ExposureAuto.set('Off')
                cam.ExposureAuto.set('On')
                exposure_time = cam.ExposureTime

                time_expo = exposure_time.get()
                print('exposure_time:', time_expo)
                inc = exposure_time.get_increment()
                # time_expo = 150000 #indoor
                # time_expo = 65000 #indoor
                # time_expo = 50000 #indoor
                time_expo = 25000 #indoor
                # time_expo = 10000 #indoor
                # time_expo = 8000 #outdoor cloudy
                # time_expo = 4000 #outdoor cloudy
                # time_expo = 2500 #outdoor cloudy
                # time_expo = 500 #outdoor cunlight
                # exposure_time.set(time + inc)
                # cam.ExposureAutoTarget.set(30)
                # cam.ExposureAuto.set('Continuous')
                exposure_time.set(time_expo)

                # self.set_nearest_value(cam, 'Height', cfg.original_size[0])
                # self.set_nearest_value(cam, 'Width', cfg.original_size[1])
                # 800 / 1500/ 1800/2000
                """
                # ground for sprayer
                self.set_nearest_value(cam, 'Height', 1616)
                # self.set_nearest_value(cam, 'Height', 2056)
                # self.set_nearest_value(cam, 'OffsetY', 0)

                self.set_nearest_value(cam, 'Width', 2260)
                self.set_nearest_value(cam, 'OffsetX', 200)
                # (2360, 2056, 3)
                """
                # ground for collection data
                # self.set_nearest_value(cam, 'Height', 1616)
                # self.set_nearest_value(cam, 'Height', 2056)

                self.set_nearest_value(cam, 'OffsetY', 0)
                self.set_nearest_value(cam, 'Height', 1850)
                # self.set_nearest_value(cam, 'Height', 1900)

                # self.set_nearest_value(cam, 'Width', 2352)
                # self.set_nearest_value(cam, 'OffsetX', 100)
                self.set_nearest_value(cam, 'OffsetX', 0)
                self.set_nearest_value(cam, 'Width', 2464)
                # self.set_nearest_value(cam, 'OffsetX', 100)
                # self.set_nearest_value(cam, 'Width', 2360)

                # soil bin 
                # self.set_nearest_value(cam, 'Width', 2264)
                # self.set_nearest_value(cam, 'OffsetX', 200)

                # AcquisitionFrameRate = cam.AcquisitionFrameRate
                # print('AcquisitionFrameRate:', AcquisitionFrameRate)
                # TriggerSelector = cam.get_feature_by_name("TriggerSelector")
                # print('TriggerSelector:', TriggerSelector)
                # TriggerMode = cam.get_feature_by_name("TriggerMode")
                # TriggerSource = cam.get_feature_by_name("TriggerSource")
                # print('TriggerMode:', TriggerMode)
                # # feature = cam.get_feature_by_name("AcquisitionFrameRate")
                # AcquisitionFrameRate.set(30) #specifies 30FPS
                # # set the other features TriggerSelector and TriggerMode
                # TriggerSelector.set("FrameStart")
                # TriggerMode.set("Off")
                # FixedRate=30
                # TriggerSource.set(FixedRate)
            except Exception as e:
                # except (AttributeError, VimbaFeatureError):
                print(e)
                pass

            # Enable white balancing if camera supports it
            try:
                cam.BalanceWhiteAuto.set('Continuous')
            except Exception as e:
                # except (AttributeError, VimbaFeatureError):
                print(e)
                pass
            # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
            try:
                stream = cam.get_streams()[0]
                stream.GVSPAdjustPacketSize.run()
                while not stream.GVSPAdjustPacketSize.is_done():
                    pass
            # except (AttributeError, VmbFeatureError):
            except Exception as e:
                pass

            self.setup_pixel_format(cam, window)
            # Query available, open_cv compatible pixel formats
            # prefer color formats over monochrome formats
            # try:
            #     cv_fmts = intersect_pixel_formats(cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
            #     color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)

            #     if color_fmts:
            #         cam.set_pixel_format(color_fmts[0])
            #         print(f'Camera set_pixel_format: {color_fmts[0]}.')
            #         if window:
            #             window.statusBar().showMessage(f'Camera set_pixel_format: {color_fmts[0]}.')
            #     else:
            #         mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)

            #         if mono_fmts:
            #             cam.set_pixel_format(mono_fmts[0])
            #         else:
            #             if window:
            #                 window.statusBar().showMessage('Camera does not support a OpenCV compatible format natively. Abort.')
            #     success, img = self.capture()
            # except Exception as e:
            #     print(e)
            #     pass


    def setup_pixel_format(self, cam, window):
        # Query available pixel formats. Prefer color formats over monochrome formats
        cam_formats = cam.get_pixel_formats()
        cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
        convertible_color_formats = tuple(f for f in cam_color_formats
                                        if opencv_display_format in f.get_convertible_formats())

        cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
        convertible_mono_formats = tuple(f for f in cam_mono_formats
                                        if opencv_display_format in f.get_convertible_formats())

        # if OpenCV compatible color format is supported directly, use that
        if opencv_display_format in cam_formats:
            cam.set_pixel_format(opencv_display_format)

        # else if existing color format can be converted to OpenCV format do that
        elif convertible_color_formats:
            cam.set_pixel_format(convertible_color_formats[0])

        # fall back to a mono format that can be converted
        elif convertible_mono_formats:
            cam.set_pixel_format(convertible_mono_formats[0])

        else:
            if window:
                # abort('Camera does not support an OpenCV compatible format. Abort.')
                window.statusBar().showMessage('Camera does not support an OpenCV compatible format. Abort.')


    def check_camera(self):
        window = None
        if cfg is not None:
            self.window = cfg.global_window
            window = self.window
        with Vimba.get_instance() as vimba:
            # interfaces = vimba.get_all_interfaces()
            # for interface in interfaces:
            #     for feat in interface.get_all_features():
            #         print(feat)
            with vimba.get_all_cameras()[0] as camera:
                self.setup_camera(camera)
                if window is not None:
                    if not camera:
                        window.statusBar().showMessage('No accessible cameras. Abort.')
                    else:
                        window.statusBar().showMessage("Camera is ready for access")
                    # self.cam = cam

    def set_exposure(self, window):
        if window:
            if window.Exposure_offBox.isChecked():
                print("off")
                window.ExpState = "Off"
            elif window.Exposure_onceBox.isChecked():
                window.ExpState = "Once"
            elif window.Exposure_continueBox.isChecked():
                print("set exposure continues")
                window.ExpState = "Continuous"
            else:
                pass


def demo():
    import numpy as np
    camera = CameraVimba()
    
    camera.check_camera()
    camera.setup_camera()
    with Vimba.get_instance() as vimba:
        with vimba.get_all_cameras()[0] as cam:
            cam.start_streaming(handler=handler, buffer_count=10)
            while True:
                start_time_tot = time()
                camera.capture()
                end_time_tot = time()
                cur_tot_fps = np.round(1.0 / (end_time_tot - start_time_tot), 2)
                print('cur_tot_fps:', cur_tot_fps)


def main():
    demo()


if __name__ == '__main__':
    main()

# import numpy as np
# depth = frames.get_depth_frame()
# depth_data = depth.as_frame().get_data()
# np_image = np.asanyarray(depth_data)
