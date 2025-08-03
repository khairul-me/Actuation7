#include <SoftwareSerial.h>
/* www.learningbuz.com */
/*Impport following Libraries*/
#include <Wire.h> 
// 12 relays
//I2C pins declaration

// SoftwareSerial mySensor1 (0, 1); // RX, TX
// const int relayPin1 = 3;
//relayPin2 -> relay3
//relayPin13 -> relay1
//relayPin9 -> relay2
const int relayPin2 = 4;
const int relayPin3 = 5;
const int relayPin4 = 6;
const int relayPin5 = 7;
const int relayPin6 = 8;
const int relayPin7 = 9;
const int relayPin8 = 10;
// const int relayPin9 = 11;
const int relayPin9 = 14; // A0
const int relayPin10 = 12;
const int relayPin11 = 13;

const int relayPin12 = 2;
const int relayPin13 = 3;

const int open_time = 2000;

char buffer[3];
int sp_quality = 0;

void setup() {
  // pinMode(relayPin1, OUTPUT);
  // digitalWrite(relayPin1, LOW);
  // delay(10);

  pinMode(relayPin2, OUTPUT);
  digitalWrite(relayPin2, LOW);
  delay(10);

  pinMode(relayPin3, OUTPUT);
  digitalWrite(relayPin3, LOW);
  delay(10);

  pinMode(relayPin4, OUTPUT);
  digitalWrite(relayPin4, LOW);
  delay(10);

  pinMode(relayPin5, OUTPUT);
  digitalWrite(relayPin5, LOW);
  delay(10);

  pinMode(relayPin6, OUTPUT);
  digitalWrite(relayPin6, LOW);
  delay(10);

  pinMode(relayPin7, OUTPUT);
  digitalWrite(relayPin7, LOW);
  delay(10);

  pinMode(relayPin8, OUTPUT);
  digitalWrite(relayPin8, LOW);
  delay(10);

  pinMode(relayPin9, OUTPUT);
  digitalWrite(relayPin9, LOW);
  delay(10);

  pinMode(relayPin10, OUTPUT);
  digitalWrite(relayPin10, LOW);
  delay(10);

  pinMode(relayPin11, OUTPUT);
  digitalWrite(relayPin11, LOW);
  delay(10);

  pinMode(relayPin12, OUTPUT);
  digitalWrite(relayPin12, LOW);
  delay(10);

  pinMode(relayPin13, OUTPUT);
  digitalWrite(relayPin13, LOW);
  delay(10);

  Serial.begin(9600);
  while (!Serial) {
    ;
  }
}


void loop() {
  delay(10);
  // digitalWrite(relayPin12, HIGH);
  // delay(open_time);
  // digitalWrite(relayPin12, LOW);
  // delay(open_time);

  if (Serial.available() > 0) {
    // much faster with length=1 than 2
    int size = Serial.readBytesUntil('e', buffer, 1);
    if (buffer[0] == '1') {
      digitalWrite(relayPin9, HIGH);
    }
    if (buffer[0] == '2') {
      digitalWrite(relayPin2, HIGH);
    }
    if (buffer[0] == '3') {
      digitalWrite(relayPin3, HIGH);
    }
    if (buffer[0] == '4') {
      digitalWrite(relayPin4, HIGH);
    }
    if (buffer[0] == '5') {
      digitalWrite(relayPin5, HIGH);
    }
    if (buffer[0] == '6') {
      digitalWrite(relayPin6, HIGH);
    }
    if (buffer[0] == '7') {
      digitalWrite(relayPin7, HIGH);
    }
    if (buffer[0] == '8') {
      digitalWrite(relayPin8, HIGH);
    }
    if (buffer[0] == '9') {
      digitalWrite(relayPin13, HIGH);
    }
    if (buffer[0] == 'a') {
      digitalWrite(relayPin10, HIGH);
    }
    if (buffer[0] == 'b') {
      digitalWrite(relayPin11, HIGH);
    }
    if (buffer[0] == 'c') {
      digitalWrite(relayPin12, HIGH);
    }
    delay(50);
    if (buffer[0] == '0') {
      // digitalWrite(relayPin1, LOW);
      // delay(5);
      digitalWrite(relayPin2, LOW);
      delay(5);
      digitalWrite(relayPin3, LOW);
      delay(5);
      digitalWrite(relayPin4, LOW);
      delay(5);
      digitalWrite(relayPin5, LOW);
      delay(5);
      digitalWrite(relayPin6, LOW);
      delay(5);
      digitalWrite(relayPin7, LOW);
      delay(5);
      digitalWrite(relayPin8, LOW);
      delay(5);
      digitalWrite(relayPin9, LOW);
      delay(5);
      digitalWrite(relayPin10, LOW);
      delay(5);
      digitalWrite(relayPin11, LOW);
      delay(5);
      digitalWrite(relayPin12, LOW);
      delay(5);
      digitalWrite(relayPin13, LOW);
    }
  }
}
