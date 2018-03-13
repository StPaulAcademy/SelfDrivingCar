# -*- coding: utf-8 -*-
"""
Created By Daniel Ellis and Michael Hall
"""

import numpy as np
from picamera import PiCamera
import socket
import time
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor

class Output(object):
    def write(self, buf):
        global ready
        global image
        y_data = np.frombuffer(buf, dtype=np.uint8, count= 160*160).reshape((160,160))
        if ready == True:
            image = y_data[:160, :160]
            ready= False
            
    def flush(self):
        pass

class Car():
    def __init__(self, host, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((str(host), port))
        self.motor_hat = Adafruit_MotorHAT(addr=0x60)
        self.motor = self.motor_hat.getMotor(1)
        self.motor.setSpeed(250)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        self.servo = GPIO.PWM(17, 50)
        self.servo.start(7.5)
        self.camera = PiCamera(sensor_mode=4, resolution='160x160', framerate=40)
        self.image = np.zeros((160,160))
        self.y_data = np.empty((160,160), dtype=np.uint8)
        time.sleep(2)
        self.output = Output()
        self.camera.start_recording(self.output, 'yuv')
        self.ready = True
        
        
    def recieveData(self, number):
        data = self.client.recv(2048)
        
        if data == b'15':
            if ready == False:
                    np.save(str(number) + ".npy", np.array([image, np.array([0,1,0])]))
            self.motor.setSpeed(200)
            self.motor.run(Adafruit_MotorHAT.FORWARD)
            self.servo.ChangeDutyCycle(7.5)
            
        if data == b'8':
            if ready == False:
                    np.save(str(number) + ".npy", np.array([image, np.array([0,0,1])]))
            self.motor.setSpeed(200)
            self.motor.run(Adafruit_MotorHAT.FORWARD)
            self.servo.ChangeDutyCycle(2.5)
            
        if data == b'14':
            if ready == False:
                    np.save(str(number) + ".npy", np.array([image, np.array([1,0,0])]))
            self.motor.setSpeed(200)    
            self.motor.run(Adafruit_MotorHAT.FORWARD)
            self.servo.ChangeDutyCycle(12.5)
        if data == b'3':
            self.motor.setSpeed(200)
            self.motor.run(Adafruit_MotorHAT.BACKWARD)
            self.servo.ChangeDutyCycle(7.5)
            
        if data == b'4':
            self.motor.setSpeed(200)
            self.motor.run(Adafruit_MotorHAT.BACKWARD)
            self.servo.ChangeDutyCycle(2.5)
            
        if data == b'2':
            self.motor.setSpeed(200)    
            self.motor.run(Adafruit_MotorHAT.BACKWARD)
            self.servo.ChangeDutyCycle(12.5)
            
        if data == b'20':
            self.client.close()
            self.motor.setSpeed(0)
            self.motor.run(Adafruit_MotorHAT.RELEASE)
            self.servo.ChangeDutyCycle(7.5)
            print("Server closed")
            self.camera.stop_recording()
            exit()
            
        if data == b'0':
            self.motor.setSpeed(0)
            self.servo.ChangeDutyCycle(7.5)
            
        else:
            pass

carl = Car()
number = 0
while True:
    carl.recieveData(number)
    number += 1
