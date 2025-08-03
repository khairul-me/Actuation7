# First import the library

class CameraBase:
    def __init__(self) -> None:
        self.is_streaming = False
        pass

    def capture(self):
        pass

    def end(self):
        pass

    def config(self):
        pass

    def camera_streaming(self, queue):
        pass
    
    def check_camera(self):
        pass

    def setup_camera(self):
        pass

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