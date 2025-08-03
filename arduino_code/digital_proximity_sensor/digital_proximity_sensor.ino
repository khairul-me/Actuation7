// https://github.com/guillaume-rico/SharpIR
// https://hackaday.com/2009/01/05/parts-digital-proximity-sensor-sharp-gp2y0d02/

const int proximity_sensor_pin = 9;
const int pneumatic_sensor_pin = 8;
int proximity_sensor_state = 0;

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT); 
  pinMode(proximity_sensor_pin, INPUT);
  pinMode(pneumatic_sensor_pin, OUTPUT);

  digitalWrite(pneumatic_sensor_pin, LOW);

  Serial.println("Setup complete");
}

void loop() {
  // Turn on the relay (activating the solenoid valve)
  proximity_sensor_state = digitalRead(proximity_sensor_pin);
  if(proximity_sensor_state==HIGH)
  {
    Serial.println(proximity_sensor_state);
    digitalWrite(LED_BUILTIN, HIGH);
    digitalWrite(pneumatic_sensor_pin, HIGH);
    // delay(2000);
    // digitalWrite(pneumatic_sensor_pin, LOW);
  }
  else
  {
    // digitalWrite(LED_BUILTIN, LOW);
  }

  delay(1);
}

