#include <Adafruit_MotorShield.h>
#include <Servo.h>
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 
Adafruit_DCMotor *myMotor = AFMS.getMotor(1);
Servo servo1;
int command = 0;

void setup() {
  AFMS.begin();
  myMotor->setSpeed(0);
  servo1.attach(9);
  Serial.begin(9600);
  Serial.setTimeout(100);
}

void loop() {
  if (Serial.available() > 0){
    command = Serial.parseInt();
    do_commands(command);
    }  
  }

void forward(){
  myMotor->setSpeed(255);
  myMotor->run(FORWARD);
  }
  
void backward(){
  myMotor->setSpeed(255);
  myMotor->run(BACKWARD);
  }

void right(){
  servo1.write(45);
  }

void left(){
  servo1.write(135);
  }

void center(){
  servo1.write(90);
  }

void stopped(){
  myMotor->setSpeed(0);
  servo1.write(90);
  }

void do_commands(int command){
  switch (command){
    case 0: stopped(); break;
    case 1: forward(); break;
    case 2: backward(); break;
    case 3: right(); break;
    case 4: left(); break;
    case 5: center(); break;
    }
  }
