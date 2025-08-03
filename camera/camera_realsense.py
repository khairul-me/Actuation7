# First import the library
from camera_abstract import CameraBase
import pyrealsense2 as rs
import numpy as np
import config as cfg


class RealSense(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = rs.pipeline()
        self.pipeline.start()    

    def capture(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        color_data = color.as_frame().get_data()
        np_image = np.asanyarray(color_data)
        np_image = np_image[:,:, ::-1]
        # depth = frames.get_depth_frame()
        # if not depth: 
        #     continue
        return True, np_image

    def end(self):
        self.pipeline.stop()

    def camera_streaming(self, queue):
        self.is_streaming = True
        print("streaming started")
        cfg.global_window.statusBar().showMessage("streaming started")
        while self.is_streaming:
            frame = None
            try:
                res, frame = self.capture()
                if not res:
                    print('capture frame failed.')
            except Exception as e:
                print(e)
            if queue.full():
                with queue.mutex:
                    queue.queue.clear()
            if frame is not None:
                queue.put(frame)
        print("streaming stopped")
        cfg.global_window.statusBar().showMessage("streaming stopped")

    def check_camera(self):
        # cfg.global_window.statusBar().showMessage('No Cameras accessible. Abort.')
        cfg.global_window.statusBar().showMessage("Camera is ready for access")

    def setup_camera(self):
        pass

def demo():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    prof = pipeline.start()
    s = prof.get_device().query_sensors()[1]
    s.set_option(rs.option.exposure, 270)
    s.set_option(rs.option.contrast, 55)
    s.set_option(rs.option.saturation, 60)
    s.set_option(rs.option.sharpness, 70)
    s.set_option(rs.option.white_balance, 5800)
    try:
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            if not depth: 
                continue

            # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
            coverage = [0]*64
            for y in range(480):
                for x in range(640):
                    dist = depth.get_distance(x, y)
                    if 0 < dist and dist < 1:
                        coverage[x//10] += 1

                if y%20 is 19:
                    line = ""
                    for c in coverage:
                        line += " .:nhBXWW"[c//25]
                    coverage = [0]*64
                    print(line)

    finally:
        pipeline.stop()

def main():
    demo

if __name__ =='__main__':
    main()

# import numpy as np
# depth = frames.get_depth_frame()
# depth_data = depth.as_frame().get_data()
# np_image = np.asanyarray(depth_data)