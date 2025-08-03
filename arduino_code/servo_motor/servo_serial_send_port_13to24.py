import serial 
import time
'''
ls /dev/ttyACM*
sudo chmod a+rw /dev/ttyACM0
python arduino_code/valve_control/valve_serial_send_port1.py 
_code/valve_control/valve_serial_send_port1.py 
'''
"""
"""
# with serial.Serial('/dev/ttyACM0', 9600, timeout=1) as ser:
with serial.Serial('COM7', 9600, timeout=1) as ser:
    first_xxx = ''
    print('serial send')
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode("ascii")
            print(data)
        if ser.out_waiting == 0:
            with open('./arduino_code/servo_motor/signal_13to24.txt', 'r') as f:
                content = f.read()
                if content == '':
                    continue
            xxx = content[0]
            val = str(int(content[2:]))
            if xxx == '0':
                ser.write(bytes('0'+'-'+val, 'utf-8'))
            if xxx == '1':
                ser.write(bytes('1'+'-'+val, 'utf-8'))
            if xxx == '2':
                ser.write(bytes('2'+'-'+val, 'utf-8'))
            if xxx == '3':
                ser.write(bytes('3'+'-'+val, 'utf-8'))
            if xxx == '4':
                ser.write(bytes('4'+'-'+val, 'utf-8'))
            if xxx == '5':
                ser.write(bytes('5'+'-'+val, 'utf-8'))
            if xxx == '6':
                ser.write(bytes('6'+'-'+val, 'utf-8'))
            if xxx == '7':
                ser.write(bytes('7'+'-'+val, 'utf-8'))
            if xxx == '8':
                ser.write(bytes('8'+'-'+val, 'utf-8'))
            if xxx == '9':
                ser.write(bytes('9'+'-'+val, 'utf-8'))
            if xxx == 'a':
                ser.write(bytes('a'+'-'+val, 'utf-8'))
            if xxx == 'b':
                ser.write(bytes('b'+'-'+val, 'utf-8'))
            if xxx == 'c':
                ser.write(bytes('c'+'-'+val, 'utf-8'))
            print('input:', xxx)
            if xxx != '':
                with open('./arduino_code/servo_motor/signal_13to24.txt', 'w') as f:
                    res = f.write('')
                    f.flush()
        time.sleep(0.01)
        # print(ser.readline())