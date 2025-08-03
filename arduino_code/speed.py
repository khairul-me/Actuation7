import numpy as np
wheel_radius = 4.5  #com
wheel_perimeter = np.pi * 2 * wheel_radius

def receive_magnetic_signal():
    with open('speed.txt', 'r') as f:
        ave_count_by_time = f.read()
        ave_count_by_time = float(ave_count_by_time)
    return ave_count_by_time

import time
while True:
    time.sleep(1)
    try:
        ave_count_by_time = receive_magnetic_signal()
        speed = ave_count_by_time * wheel_perimeter
        speed = round(speed, 3)
        print('speed:', speed)
    except:
        continue