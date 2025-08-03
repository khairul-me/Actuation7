#include <SoftwareSerial.h>

SoftwareSerial mySensor1 (10, 11); // RX, TX
int relay_pin=2;
int hall_pin=3;
int state = 0;

int allow_start = 0;
const int average_cnt = 5;
const int min_interval_milisecond = 300;
int rotate_count = 0;
float rotate_speed = 0.0;
unsigned long  t0 = 0;
unsigned long  t1 = 0;
unsigned long  t_ref0 = 0;
unsigned long  t_ref1 = 0;
unsigned long  t_ref_stop = 0;

char buffer[3];
unsigned long times[average_cnt];
  
void setup() {
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT); 
  pinMode(relay_pin, OUTPUT);
  digitalWrite(relay_pin, HIGH);
  Serial.begin(9600);
  //mySensor1.begin(12500);
  //Serial.println("start! ");
  //Log.begin(LOG_LEVEL_VERBOSE, &Serial, true);
  while (!Serial) {
    ; // wait for serial port to connect.
  }
  t_ref0 = millis();
}
float calculate_speed(int rotate_count, int average_cnt)
{
  rotate_speed = 0;
  if (rotate_count == average_cnt)
  {
    t0 = times[rotate_count-average_cnt];
    t1 = times[rotate_count-1];
    if (t1 > t0)
    {
      rotate_speed  = float(average_cnt-1)*1000.0/(float(t1) - float(t0));
    }
  }
  else if (rotate_count == 0)
  {
    t0 = times[average_cnt-1];
    t1 = times[rotate_count];
    if (t1 > t0)
    {
      rotate_speed  = float(average_cnt-1)*1000.0/(float(t1) - float(t0));
    }
  }
  else if (rotate_count > 0)
  {
    t0 = times[rotate_count];
    t1 = times[rotate_count-1];
    if (t1 > t0)
    {
      rotate_speed  = float(average_cnt-1)*1000.0/(float(t1) - float(t0));
    }
  }
  return rotate_speed;
}

void loop() {
  //Serial.println("start! ");
  delay(1); 
  // put your main code here, to run repeatedly:
  state = digitalRead(hall_pin);
  t_ref1 = millis();
  if(state==LOW)
  {
    digitalWrite(LED_BUILTIN, HIGH);
    if (allow_start==1 && rotate_count==1)
    {
      // t0 = millis();
      // Serial.print("start ");
      // Serial.print(rotate_count);
      // Serial.print(" t0 ");
      // Serial.println(t0);
      allow_start = 0;
    }
    else if (rotate_count==average_cnt)
    {
      // t1 = millis();
      // Serial.print("end ");
      // Serial.print(rotate_count);
      // Serial.print(" t1 ");
      // Serial.print(t1);
      // rotate_speed  = float(average_cnt)*1000.0/(float(t1) - float(t0));
      // Serial.print(" rotate_speed ");
      // Serial.println(rotate_speed);
      rotate_count=0;
    }
    if(t_ref1 - t_ref0 > min_interval_milisecond)
    {
      // Serial.print("t_ref0 ");
      // Serial.print(t_ref0);
      // Serial.print(" t_ref1 ");
      // Serial.print(t_ref1);
      // Serial.print(" t_ref1 - t_ref0 ");
      // Serial.print(t_ref1 - t_ref0);
      t_ref0 = millis();
      rotate_count +=1;
      // Serial.print(" rotate_count ");
      // Serial.println(rotate_count);
      allow_start = 1;
      times[rotate_count-1] = t_ref0;
      // Serial.println(rotate_count);
      rotate_speed = calculate_speed(rotate_count, average_cnt);
      Serial.println(rotate_speed);
      // Serial.print(' ');
      // for(int i= 0; i<average_cnt; ++i)
      // {
      //   Serial.print(times[i]);
      //   Serial.print(' ');
      // }
      // Serial.print(rotate_count);
      // Serial.println(' ');

      t_ref_stop = millis();
    }
  }
  else
  {
    digitalWrite(LED_BUILTIN, LOW);
  }
  if(t_ref1 - t_ref_stop > 3000)
  {
    t_ref_stop = millis();
    if(rotate_count>0)
    {
      t1 = rotate_count-1;
      times[t1] = millis();
    }
    else
    {
      t1 = average_cnt - 1;
      times[t1] = millis();
    }
    rotate_speed = calculate_speed(rotate_count, average_cnt);

    Serial.println(rotate_speed);
    // Serial.print(' ');
    // for(int i= 0; i<average_cnt; ++i)
    // {
    //   Serial.print(times[i]);
    //   Serial.print(' ');
    // }
    // Serial.print(rotate_count);
    // Serial.println(' ');
  }

  //Log.verboseln("xxx");
  // if we get a command, turn the LED on or off:
  if (Serial.available() > 0) {
    int size = Serial.readBytesUntil('e', buffer, 1);
    //int incomingByte = Serial.read();
    //Log.verboseln(buffer);
    //Serial.println(buffer);
    if (buffer[0] != 'x' && buffer[0] != 'N') {
      digitalWrite(relay_pin, LOW);
    }
    if (buffer[0] == 'N') {
      digitalWrite(relay_pin, HIGH);
    }
    //buffer[0] = "x";
    //buffer[1] = "x";
    //buffer[2] = "x";
  }
}
