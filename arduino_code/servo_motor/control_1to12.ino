#include <Servo.h>

Servo servos[12];
const int servoPins[12] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
int servoOffsets[12] = {0, 5, 3, -2, -5, 0, 0, -5, 5, -8, 0, -5}; // Offsets per servo
const int operation_interval = 500;

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 12; i++) {
    servos[i].attach(servoPins[i]);
  }
  for (int i = 0; i < 12; i++) {
    int angle = 90 + servoOffsets[i];
    angle = constrain(angle, 35, 145);
    servos[i].write(angle);
    delay(5);
  }
  Serial.println("All servos reset to default positions.");
  Serial.println("Setup complete");
}

int getServoIndex(char id) {
  if (id >= '1' && id <= '9') return id - '1';       // '1' to '9' → 0 to 8
  if (id == 'a') return 9;
  if (id == 'b') return 10;
  if (id == 'c') return 11;
  return -1; // invalid
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim(); // remove whitespace

    int start = 0;
    while (start < input.length()) {
      int end = input.indexOf(' ', start);
      if (end == -1) end = input.length();

      String token = input.substring(start, end);
      int dashIndex = token.indexOf('-');

      // "0" without dash means reset all
      if (dashIndex == -1 && token == "0") {
        for (int i = 0; i < 12; i++) {
          int angle = 90 + servoOffsets[i];
          angle = constrain(angle, 35, 145);
          servos[i].write(angle);
          delay(5);
        }
        Serial.println("All servos reset to default positions.");
      }

      // Process commands like "1-130"
      if (dashIndex != -1 && token.length() > dashIndex + 1) {
        char id = token.charAt(0);
        int angle = token.substring(dashIndex + 1).toInt();

        int index = getServoIndex(id);
        if (index == -1) {
          Serial.print("Invalid servo ID: ");
          Serial.println(id);
        } else {
          angle += servoOffsets[index];
          angle = constrain(angle, 35, 145);

          servos[index].write(angle);

          Serial.print("Servo ");
          Serial.print(id);
          Serial.print(" set to ");
          Serial.println(angle);
        }
      }

      start = end + 1;
    }

    delay(50);
  }
}
