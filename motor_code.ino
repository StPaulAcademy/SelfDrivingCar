#include <Adafruit_MotorShield.h>
#include <Servo.h>
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 
Adafruit_DCMotor *myMotor = AFMS.getMotor(1);
Servo servo1;

void setup() {
  AFMS.begin();
  myMotor->setSpeed(0);
  servo1.attach(9);
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0){
    int mtr = Serial.parseInt();
    int spd = Serial.parseInt();
    int dir = Serial.parseInt();
    int turn = Serial.parseInt();
    if (mtr == 0){
      servo1.write(turn);
      }
    if (mtr==1){
      myMotor->setSpeed(spd);
      if (dir == 1){
        myMotor->run(FORWARD);
        }
      if (dir == 0){
        myMotor->run(BACKWARD);
        }
      }
    }
}
