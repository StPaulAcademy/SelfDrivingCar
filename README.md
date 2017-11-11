# Advanced Technology Projects Fall 2017, St. Paul Academy
--------------------------------
##### Self-Driving, 3d printed car

###### By Michael Hall and Daniel Ellis '18

### Hardware

Structural components of the car are designed from scratch in Fusion 360 and 3D printed in PLA plastic on Makerbot Replicator 2 printers. 
The car is essentially a rear-wheel drive car driving backwards. 
This choice was made because after construction of the car it was discovered that the car worked much better backwards than forwards. 
For motors, the car has a servo in the back for steering and a 12v dc motor geared 1:1 on the front.
The motors are controlled using an arduino with an adafruit motorshield.
The arduino recieves commands from a Raspberry Pi 3 which is the brains of the car.

### Software

Except for the arduino which is programmed in C, all other code is in Python. 

##### Training Data

To capture training data we used a socket server to connect the Raspberry Pi to our computers and send commands from the q, w, and e keys for left, forward, and right. 
Images are captured using a PiCamera in black and white with a 160 x 90 resolution.
The images are then saved along with the label (move taken before the picture) and saved as a numpy array.
After collecting enough data the images are compiled into a single file ready for the neural net.
