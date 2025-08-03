const int relayPin = 7;

void setup() {
  Serial.begin(9600);

  // Set the relayPin as an OUTPUT
  pinMode(relayPin, OUTPUT);

  // Start with the relay turned off
  digitalWrite(relayPin, LOW);

  Serial.println("Setup complete");
}

void loop() {
  // Turn on the relay (activating the solenoid valve)
  digitalWrite(relayPin, HIGH);
  Serial.println("Relay ON");
  delay(1000);

  // Turn off the relay (deactivating the solenoid valve)
  digitalWrite(relayPin, LOW);
  Serial.println("Relay OFF");  
  delay(5000);
}

