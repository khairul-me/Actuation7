import serial 
import time
'''
ls /dev/ttyACM*
sudo chmod a+rw /dev/ttyACM0
python arduino_code/valve_control/valve_serial_send_port1.py 
_code/valve_control/valve_serial_send_port1.py 
'''
"""
1 -> relay2 -> Arduino port 14
9 -> relay1 -> Arduino port 13
"""
# with serial.Serial('/dev/ttyACM0', 9600, timeout=1) as ser:
with serial.Serial('COM4', 9600, timeout=1) as ser:
    first_xxx = ''
    print('serial send')
    while True:
        # xxx=input("LED on?")
        if ser.in_waiting > 0:
            data = ser.readline().decode("ascii")
            print(data)
            # with open('speed.txt', 'w') as f:
            #     f.write(data)

        if ser.out_waiting == 0:
            with open('./arduino_code/valve_control/signal_1to12.txt', 'r') as f:
                xxx = f.read()
                if xxx == '':
                    continue
            # if first_xxx == xxx:
            #     continue
            # else:
            #     first_xxx = xxx
            if xxx == '0':
                ser.write(bytes('0', 'utf-8'))
            if xxx == '1':
                ser.write(bytes('1', 'utf-8'))
            if xxx == '2':
                ser.write(bytes('2', 'utf-8'))
            if xxx == '3':
                ser.write(bytes('3', 'utf-8'))
            if xxx == '4':
                ser.write(bytes('4', 'utf-8'))
            if xxx == '5':
                ser.write(bytes('5', 'utf-8'))
            if xxx == '6':
                ser.write(bytes('6', 'utf-8'))
            if xxx == '7':
                ser.write(bytes('7', 'utf-8'))
            if xxx == '8':
                ser.write(bytes('8', 'utf-8'))
            if xxx == '9':
                ser.write(bytes('9', 'utf-8'))
            if xxx == '10':
                ser.write(bytes('a', 'utf-8'))
            if xxx == '11':
                ser.write(bytes('b', 'utf-8'))
            if xxx == '12':
                ser.write(bytes('c', 'utf-8'))
            print('input:', xxx)
            if xxx != '':
                with open('./arduino_code/valve_control/signal_1to12.txt', 'w') as f:
                    res = f.write('')
                    f.flush()
        time.sleep(0.01)
        # print(ser.readline())
