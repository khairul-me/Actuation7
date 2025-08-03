const int dirPin = 2;
const int pulsePin = 3;
const int step_per_rev = 3200;
const int step_for_cm = 5*step_per_rev;
const int step_for_cm_small = 1*step_per_rev;
const int step_for_cm_tiny = 0.1*step_per_rev;

void setup() {
  Serial.begin(9600);
  // put your setup code here, to run once:
  pinMode(dirPin, OUTPUT);
  pinMode(pulsePin, OUTPUT);
  digitalWrite(dirPin, LOW);

}

void set_stepper_dir(int dir){
  if(dir==0){
    digitalWrite(dirPin, LOW);
    Serial.println("dir 1");
  }
  else{
    digitalWrite(dirPin, HIGH);
    Serial.println("dir 2");
  }
}
void move_stepper(int dir, int steps)
{
  set_stepper_dir(dir);
  for(int i =0; i<steps;++i)
  {
    digitalWrite(pulsePin, HIGH);
    delayMicroseconds(200);
    digitalWrite(pulsePin, LOW);
    delayMicroseconds(200);
  }
}
void loop() {
  int val=-1;
  //Serial.println("OK");
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
    val = Serial.parseInt();
    // Serial.println(val);
    if (val==1){
      move_stepper(1, step_for_cm);
    }
    else if (val==2){
      move_stepper(0, step_for_cm);
    }
    else if (val==4){
      move_stepper(1, step_for_cm_small);
    }
    else if (val==5){
      move_stepper(0, step_for_cm_small);
    }
    else if (val==7){
      move_stepper(1, step_for_cm_tiny);
    }
    else if (val==8){
      move_stepper(0, step_for_cm_tiny);  
    }
  }
}
