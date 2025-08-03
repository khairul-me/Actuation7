#include <Servo.h>

Servo myservo1;
Servo myservo2;
Servo myservo3;
Servo myservo4;
Servo myservo5;
Servo myservo6;
Servo myservo7;
Servo myservo8;
Servo myservo9;
Servo myservo10;
Servo myservo11;
Servo myservo12;
const int operation_interval = 500;
char buffer[4];

void setup() {
  Serial.begin(9600);
  myservo1.attach(2);
  myservo2.attach(3);
  myservo3.attach(4);
  myservo4.attach(5);
  myservo5.attach(6);
  myservo6.attach(7);
  myservo7.attach(8);
  myservo8.attach(9);
  myservo9.attach(10);
  myservo10.attach(11);
  myservo11.attach(12);
  myservo12.attach(13);

  Serial.println("Setup complete");
}
void test()
{
  // int val=120;

  // while (Serial.available() > 0)
  // {
  //   val = Serial.parseInt();
  //   if (val < 45) val = 45;
  //   if (val > 135) val = 135; 
  //   if(val > 0){
  //     Serial.println(val);
  //     myservo1.write(val);
  //   }
  //   delay(15);
  // }
  // for (int pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
  //   // in steps of 1 degree
  //   myservo.write(pos);              // tell servo to go to position in variable 'pos'
  //   delay(15);                       // waits 15ms for the servo to reach the position
  // }
  int bias = 0;
  myservo1.write(45-bias);
  delay(operation_interval);
  myservo1.write(90-bias);
  delay(operation_interval);
  myservo1.write(135-bias);
  delay(operation_interval);
  myservo1.write(90-bias);
  delay(1000);
}
void loop() {
  test();
}

