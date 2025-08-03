//Parameters
const int gp2y0a21Pin  = A0;
const int relayPin = 7;

//Variables
int gp2y0a21Val  = 0;
char buffer[3];

void setup() {
  //Init Serial USB
  Serial.begin(9600);
  Serial.println(F("Initialize System"));
  //Init ditance ir
  pinMode(relayPin, OUTPUT);
  pinMode(gp2y0a21Pin, INPUT);
  digitalWrite(relayPin, LOW);

}

void loop() {
  // testGP2Y0A21();
  control_air_compressor();

}
void control_air_compressor( ) { /* function testGP2Y0A21 */
  if (Serial.available() > 0) {
    int size = Serial.readBytesUntil('e', buffer, 1);
    if (buffer[0] != 'N') {
      digitalWrite(relayPin, LOW);
    }
    if (buffer[0] == 'H') {
      digitalWrite(relayPin, HIGH);
    }
  }
}

void testGP2Y0A21( ) { /* function testGP2Y0A21 */
  ////Read distance sensor
  gp2y0a21Val = analogRead(gp2y0a21Pin);
  // Serial.print(gp2y0a21Val); Serial.print(F(" - ")); Serial.println(distRawToPhys(gp2y0a21Val));
  Serial.print(gp2y0a21Val);
  // if (gp2y0a21Val < 400 && gp2y0a21Val > 20) {
  // if (gp2y0a21Val > 350) {
  if (gp2y0a21Val > 320) {
  // if (gp2y0a21Val > 250) {
    Serial.println(F(" o detected"));
    // delay(1500);
    delay(1500);
    digitalWrite(relayPin, HIGH);
    delay(500);
    digitalWrite(relayPin, LOW);
    // delay(2000);
    delay(1000);
  } else {
    Serial.println(F(" No obstacle"));
    digitalWrite(relayPin, LOW);
  }
  // delay(150);
}

int distRawToPhys(int raw) { /* function distRawToPhys */
  ////IR Distance sensor conversion rule
  float Vout = float(raw) * 0.0048828125; // Conversion analog to voltage
  int phys = 13 * pow(Vout, -1); // Conversion volt to distance

  return phys;
}