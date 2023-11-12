# RobotFromVUB
Code to control a small robot built by the mechanics department of the VUB Vrije Universiteit Brussel. The robot is composed of a Raspbarry Pi 4B, which acts as Master, executing the code and performing 'high-level' operations, and an Arduino, which acts as Slave, controlling sensors and motors, and sending data to the Master.

The robot is one of the components of the 'FREE ENERGY' project carried out by the department of mechanics at VUB Vrije Universiteit Brussel.

The robot scans images captured by a Raspberry Pi Camera through a YOLO neural network model, trained with example objects to test and understand its potential. When it recognises an object from among those it has been trained with, it passes the bounding box to a Tracking System, which retains its position in the image and follows it in the next ones. The Tracking System allows the robot to avoid having to repeat the recognition of the object, and thus the prediction of the neural network, saving time and resources. The robot then moves towards the recognised object and grabs it with its gripper. After that, it returns to the starting point.

![20220531_130346](https://github.com/ClousTom/RobotFromVUB/assets/117213899/993b762f-d2f1-48a6-a8b5-0b63f07211a2)
