import serial 
import time

#with serial.Serial('/dev/ttyACM1', 9600, timeout=1) as ser:
with serial.Serial('COM3', 9600, timeout=1) as ser:
    first_xxx = 'X'
    print('serial send')
    while True:
        # xxx=input("LED on?")
        if ser.in_waiting > 0:
            data = ser.readline().decode("ascii")
            print(data)
            # with open('speed.txt', 'w') as f:
            #     f.write(data)

        if ser.out_waiting == 0:
            with open('signal.txt', 'r') as f:
                xxx = f.read()
            if first_xxx == xxx:
                continue
            else:
                first_xxx = xxx
            if xxx in 'lL':
                ser.write(bytes('L', 'utf-8'))
            if xxx in 'hH':
                ser.write(bytes('H', 'utf-8'))
        time.sleep(0.1)
        # print(ser.readline())
