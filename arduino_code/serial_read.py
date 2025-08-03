import serial 
import time

arduino = serial.Serial('COM5', 9600, timeout=1)
name = 'xxx'
save_path = fr'C:\Users\E-ITX\Desktop\BoyangDeng\code1\sensor_data_{name}.txt'

def log_xalg(*info, log_path=save_path, show=True, end=None, flush=False):
    if show:
        if end is not None:
            print(*info, end=end)
        else:
            print(*info)
    if log_path is not None:
        if flush:
            f_log = open(log_path, 'w', encoding="utf-8")
        else:
            f_log = open(log_path, 'a', encoding="utf-8")
            print(*info, file=f_log)
        f_log.close()
# log_xalg('', flush=True)

while True:
    try:
        time.sleep(0.01)
        if arduino.in_waiting > 0:
            data = arduino.readline().decode("utf-8")
            # if data > 320:
            #     print("Obstacle detected")
            # print('data:', data)
            log_xalg(data)
    except Exception as e:
        continue
arduino.close()
