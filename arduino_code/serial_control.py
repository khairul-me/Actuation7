import time
import serial

ser = serial.Serial('/dev/serial0', 115200, timeout=0.050)

count = 0
state = 0

ser.write("Script Started").encode())

while 1:
    if(state == 0):          #waits for incoming data
        while ser.in_waiting:
            data = ser.readline().decode("ascii")
            if(data == '1')  #received a '1' move onto next state
                state = 1;
                print("1 received")
                ser.write("1 received").encode()
            else:           #wrong data stay at state 0
                print("back to the start")
                ser.write("back to the start").encode()

    elif(state == 1):          #waits for incoming data
        while ser.in_waiting:
            data = ser.readline().decode("ascii")
            if(data == '2')  #received a '2' move on to next state
                state = 2;
                print("2 received")
                ser.write("2 received").encode()
            else:            #wrong data return to state 0  
                state = 0;
                print("back to the start")
                ser.write("back to the start").encode()

    elif(state == 2):          #waits for incoming data
        while not ser.out_waiting:
            data = ser.readline().decode("ascii")
            if(data == '3')  #received a '3'  //received a '3' print message
                print("You win!")
                ser.write("You win!").encode()
                state = 0; 
            else:             #wrong data return to state 0  
                state = 0;
                print("back to the start")
                ser.write("back to the start").encode()