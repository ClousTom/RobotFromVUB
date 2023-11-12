from cmath import pi
import time
import threading
from turtle import distance
from picamera import PiCamera
import keyboard
from io import BytesIO
import os
import numpy as np
import math
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite
import cv2
import bbox_visualizer as bbv
import tkinter as Tkinter
from tkinter import *
#import GUI

from Arduino_aansturingen import*
from Arduino_aansturingen import motoren_rechtdoor as rechtdoor
from Arduino_aansturingen import motoren_draaien_Rechts as rechts
from Arduino_aansturingen import motoren_draaien_Links as links
# from Arduino_aansturingen import batterij_check
#from Arduino_aansturingen import afstands_sensor_ultra as afstandq
#from Arduino_aansturingen import encoder_L_R
#from ENCODER import encoder_L_R


#Mode counting
steps = ["Mode Selection","Object Selection","Walking","Grabbing","Raising","Return","Return","Return","Return","Release"] #0-9, total=10
step=0


# Set up tracker.
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD','MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7] #change number to use another Tracker

def select_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker


def detect(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_tensor = input_tensor.reshape(1, 320, 320, 3)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    nr_readings = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])
    return nr_readings, boxes, classes, scores

# Output = bbox: [xmin, ymin, xmax, ymax]
def parse_bounding_box(bbox):
    temp = bbox[0]
    bbox[0] = bbox[1]
    bbox[1] = temp
    temp = bbox[2]
    bbox[2] = bbox[3]
    bbox[3] = temp
    bbox[0] = bbox[0]*640
    bbox[1] = bbox[1]*480
    bbox[2] = bbox[2]*640
    bbox[3] = bbox[3]*480
    return np.int32(bbox)

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model0902B.tflite")  # model-finaal123456.tflite
interpreter.allocate_tensors()

# Open the videocam streamer
video = cv2.VideoCapture(0)

#Global variables for Detection and Tracking System
bbox=None
frame=None
ok=None
center=[0,0]
#FPS
prev_frame_time = 0
new_frame_time = 0


#Variables and functions to find distance
frame_width = int(video.get(3))
frame_height = int(video.get(4))
def prct(percent, whole):
  return int((percent*whole)/100)

# variables for distance
fcx=int(frame_width/2)

fcy=int(frame_height/2)
liney1=[frame_height-prct(10,fcy),36]
liney2=[frame_height-prct(20,fcy),40]
liney3=[frame_height-prct(30,fcy),45]
liney4=[frame_height-prct(40,fcy),50]
liney5=[frame_height-prct(50,fcy),57]
liney6=[frame_height-prct(60,fcy),66]
liney7=[frame_height-prct(70,fcy),83]
liney8=[frame_height-prct(80,fcy),110]
liney9=[frame_height-prct(90,fcy),160]
liney10=[fcy,270]

#draw lines for distance
def writelines(frame):
    cv2.line(frame,(fcx-prct(50,fcx),liney1[0]),(fcx+prct(50,fcx),liney1[0]),(100,255,0),1)
    cv2.putText(frame, str(liney1[1])+"cm", (fcx+prct(51,fcx),liney1[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(46,fcx),liney2[0]),(fcx+prct(46,fcx),liney2[0]),(100,255,0),1)
    cv2.putText(frame, str(liney2[1])+"cm", (fcx+prct(47,fcx),liney2[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(42,fcx),liney3[0]),(fcx+prct(42,fcx),liney3[0]),(100,255,0),1)
    cv2.putText(frame, str(liney3[1])+"cm", (fcx+prct(43,fcx),liney3[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(38,fcx),liney4[0]),(fcx+prct(38,fcx),liney4[0]),(100,255,0),1)
    cv2.putText(frame, str(liney4[1])+"cm", (fcx+prct(39,fcx),liney4[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(34,fcx),liney5[0]),(fcx+prct(34,fcx),liney5[0]),(100,255,0),1)
    cv2.putText(frame, str(liney5[1])+"cm", (fcx+prct(35,fcx),liney5[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(30,fcx),liney6[0]),(fcx+prct(30,fcx),liney6[0]),(100,255,0),1)
    cv2.putText(frame, str(liney6[1])+"cm", (fcx+prct(31,fcx),liney6[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(26,fcx),liney7[0]),(fcx+prct(26,fcx),liney7[0]),(100,255,0),1)
    cv2.putText(frame, str(liney7[1])+"cm", (fcx+prct(27,fcx),liney7[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(22,fcx),liney8[0]),(fcx+prct(22,fcx),liney8[0]),(100,255,0),1)
    cv2.putText(frame, str(liney8[1])+"cm", (fcx+prct(23,fcx),liney8[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(18,fcx),liney9[0]),(fcx+prct(18,fcx),liney9[0]),(100,255,0),1)
    cv2.putText(frame, str(liney9[1])+"cm", (fcx+prct(19,fcx),liney9[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    cv2.line(frame,(fcx-prct(14,fcx),liney10[0]),(fcx+prct(14,fcx),liney10[0]),(100,255,0),1)
    cv2.putText(frame, ">"+str(liney10[1])+"cm", (fcx+prct(15,fcx),liney10[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
    return frame

def proportion(nextline,point,mnextline):
    a=nextline-fcy
    b=point-fcy
    c=point-nextline
    d=a-c
    return ((d*mnextline)/a)

def finddistance(objbase):
    if objbase[1]<liney5[0]:
        if objbase[1]<liney8[0]:
            if objbase[1]>=liney9[0]:
                return (proportion(liney9[0],objbase[1],liney9[1])+proportion(liney8[0],objbase[1],liney8[1]))/2
            elif objbase[1]<liney9[0] and objbase[1]>liney10[0]:
                return (liney10[1]+proportion(liney9[0],objbase[1],liney9[1]))/2
            elif objbase[1]<=liney10[0]:
                return liney10[1]
        else:
            if objbase[1]>=liney6[0]:
                return (proportion(liney6[0],objbase[1],liney6[1])+proportion(liney5[0],objbase[1],liney5[1]))/2
            elif objbase[1]<liney6[0] and objbase[1]>=liney7[0]:
                return (proportion(liney7[0],objbase[1],liney7[1])+proportion(liney6[0],objbase[1],liney6[1]))/2
            elif objbase[1]<liney7[0]:
                return (proportion(liney8[0],objbase[1],liney8[1])+proportion(liney7[0],objbase[1],liney7[1]))/2
    else:
        if objbase[1]<liney3[0]:
            if objbase[1]>=liney4[0]:
                return (proportion(liney4[0],objbase[1],liney4[1])+proportion(liney3[0],objbase[1],liney3[1]))/2
            elif objbase[1]<liney4[0]:
                return (proportion(liney5[0],objbase[1],liney5[1])+proportion(liney4[0],objbase[1],liney4[1]))/2
        else:
            if objbase[1]>=liney1[0]:
                return proportion(liney1[0],objbase[1],liney1[1])
            elif objbase[1]<liney1[0] and objbase[1]>=liney2[0]:
                return (proportion(liney2[0],objbase[1],liney2[1])+proportion(liney1[0],objbase[1],liney1[1]))/2
            elif objbase[1]<liney2[0]:
                return (proportion(liney3[0],objbase[1],liney3[1])+proportion(liney2[0],objbase[1],liney2[1]))/2


# Raspberrypi Encoder
BUTTON_GPIOAL = 5
BUTTON_GPIOBL = 6
BUTTON_GPIOAR = 23
BUTTON_GPIOBR = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIOAL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_GPIOBL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_GPIOAR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_GPIOBR, GPIO.IN, pull_up_down=GPIO.PUD_UP)


#Variables
i = 0
pot_value = []   # waarde van de potentiometer wordt opgeslagen in een list
# Lengte_1 = 0.15  #lengte draad grijper in rust
# Lengte_2 = 0.15  # Wanneer de servo draait zal dit veranderen, maar in het begin start deze op 0.15 m
straal = 15  # in milimeter 
positie = [0, 0, 0]     # [ hoek, x-coord, y-coord] Om bij te houden waar de robot zich bevindt
positie_old = [0, 0, 0]
# Circumference_wheel = 214
mm_per_puls = 0.93
afstand_wielen = 410 # afstand van 1 wiel tot de andere. Is eiglk de straal van de robot
pulsen_list = [0,0]   #[Links, Rechts]
# CounterL = 0

pos_x = []
pos_y = []
pos_hoek = []
tijden = {}

#pos_x.append(0)
#pos_y.append(0)
#pos_hoek.append(0)    


User_input_object = None
User_input_mode = None


#Say "Hello"    
time.sleep(2) #Don't take off to avoid the robot gets stuck
wrist_manual(100)
time.sleep(0.8)
wrist_manual(51)

"""
automatic_gui = False
manual_gui = False
Backwards_gui = False
Forward_gui = False

#GUI
def GUI():
    global User_input_mode
    app = Tkinter.Tk()

    #Set the geometry
    app.geometry("1000x1000")

    def which_button(button_press):
        # Printing the text when a button is clicked
        print(button_press)

    def Manual():
        print("Manual mode from GUI activated")
        global User_input_mode
        global manual_gui
        manual_gui = True
        User_input_mode=int(2)
        print("dit i de userinput", User_input_mode)
        return User_input_mode
        # return man_gui

    def Automatic():
        print("Automatic mode from GUI activated")
        auto_gui = True
        User_input_mode=0
        return User_input_mode
        # return auto_gui

    def Sequential():
        print("Sequential mode from GUI activated")
        seq_gui = True
        User_input_mode=3
        #return seq_gui
        return User_input_mode

    def Forward():
        print("Forward mode from GUI activated")
        global Forward_gui
        Forward_gui = True
        time.sleep(0.1)
        Forward_gui=False
        return Forward_gui

    def Backward():
        print("Backward mode from GUI activated")
        global Backwards_gui
        Backwards_gui = True
        time.sleep(0.1)
        Backwards_gui = False
        return Backwards_gui

    # Creating and displaying of buttons
    b1 = Button(app, text="Forward",
                command=lambda m="We gaan rechtdoor": Forward())
    b1.place(x= 300, y=100)

    b2 = Button(app, text="Backwards",
                command=lambda m="We gaan achterwaarts": Backward())
    b2.place(x= 290, y=250)

    b3 = Button(app, text="Left",
                command=lambda m="We gaan rechtdoor": which_button(m))
    b3.place(x= 225, y=175)

    b4 = Button(app, text="Right",
                command=lambda m="We gaan achterwaarts": which_button(m))
    b4.place(x=400, y=175)

    b5 = Button(app, text="UP",
                command=lambda m="We gaan achterwaarts": which_button(m))
    b5.place(x=300, y=375)

    b6 = Button(app, text="DOWN",
                command=lambda m="We gaan achterwaarts": which_button(m))
    b6.place(x=290, y=475)

    b7 = Button(app, text="GRIP",
                command=lambda m="We gaan achterwaarts": which_button(m))
    b7.place(x=300, y=575)

    b8 = Button(app, text="RELEASE",
                command=lambda m="We gaan achterwaarts": which_button(m))
    b8.place(x=290, y=650)

    canvas1 = Canvas(app, width = 220, height = 230)
    canvas1.place(x=500, y=70)
    img1 = PhotoImage(file="robot_rijden (1).png")
    canvas1.create_image((20,20), anchor=NW, image=img1)

    #  UP / DOWN arm
    canvas2 = Canvas(app, width = 270, height = 160)
    canvas2.place(x=500, y=370)
    img2 = PhotoImage(file="robot_arm (1).png")
    canvas2.create_image((20,20), anchor=NW, image=img2)

    #  Grip / Release arm
    canvas3 = Canvas(app, width = 170, height = 140)
    canvas3.place(x=500, y=520)
    img3 = PhotoImage(file="robot_grijper (1).png")
    canvas3.create_image((20,20), anchor=NW, image=img3)

    # Creating and displaying of Mode buttons
    b10 = Button(app, text="Manual",
            command=lambda m="We gaan rechtdoor": Manual())
    b10.place(x=100, y=100)
    
    b11 = Button(app, text="Automatic",
                command=lambda m="We gaan achterwaarts": Automatic())
    b11.place(x=90, y=150)

    b12 = Button(app, text="Sequential",
                command=lambda m="We gaan achterwaarts": Sequential())
    b12.place(x=90, y=200)

    app.mainloop()
                            #In het begin wordt dus gevraagd wat de working mode zal zijn      Dit wordt dan op een schermpje geprint op de robot

gui = threading.Thread(target=GUI)
gui.start()
"""

"""
# LEFT-HAND ENCODER FUNCTION
def encoder_L():
    BUTTON_GPIOAL = 5
    BUTTON_GPIOBL = 6
    #BUTTON_GPIOAR = 23
    #BUTTON_GPIOBR = 24
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_GPIOAL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(BUTTON_GPIOBL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    #GPIO.setup(BUTTON_GPIOAR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    #GPIO.setup(BUTTON_GPIOBR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    Last_State_AL = GPIO.input(5)
    #print("dit is de laatste state van AL", Last_State_AL)
    CounterL = 0
    #Last_State_AR = GPIO.input(23)
    #CounterR = 0

    while True:
        #Last_State_AL = GPIO.input(5)
        GPIO.wait_for_edge(BUTTON_GPIOBL, GPIO.RISING)

        State_AL = GPIO.input(5)
        #print("dit is de state van AL", State_AL)
        #State_AR = GPIO.input(23)

        # DEEL DAT DE PULSEN TELT VAN LINKERWIEL
        #print("dit is gpio6", GPIO.input(6))

        if State_AL != Last_State_AL:
            #if GPIO.input(6) != State_AL:
            CounterL = CounterL + 1
                #print("WE GAAN NAAR VOOOOOOOORR")
        else:
            CounterL = CounterL - 1
                #print(" WE GAAAANNNN NAAAARR ACHTEEERRRR")

        #print("We zijn aan",CounterL,"pulsen")
        pulsen_list[0] = CounterL
"""

"""
#RIGHT-HAND ENCODER FUNCTION
def encoder_R():
    while True:
        Last_State_AR = GPIO.input(23)
        CounterR = 0
        GPIO.wait_for_edge(BUTTON_GPIOAR, GPIO.RISING)
        State_AR = GPIO.input(23)
        #PART THAT COUNTS PULSES OF RIGHT WHEEL
        if State_AR != Last_State_AR:
            if GPIO.input(24) != State_AR:
                CounterR = CounterR + 1
                #print("WE GAAN NAAR VOOOOOOOORR")
            else:
                CounterR = CounterR - 1
                #print(" WE GAAAANNNN NAAAARR ACHTEEERRRR")
        #print("We are at ",CounterR," pulses.\n")
        pulsen_list[1] = CounterR
        #return pulsen_list
"""

# Visual, Detection and Tracking System
def visual_camera():
    global bbox, center, frame, User_input_object,User_input_mode,prev_frame_time,new_frame_time,step
    def detection(bbox1,frame):
        # Load test images and resize the input size for the model 320x320 pixels
        global User_input_object
        result, frame = video.read()
        if not result:
            print("Error")
        img = cv2.resize(frame, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
        img = np.float32(img/255)

        input_tensor = img
        nr_readings, boxes, classes, scores = detect(interpreter, input_tensor)
        for j in range(int(nr_readings[0])):
            if scores[0][j] > 0.8:
                # print("detection", scores[0][j])
                bbox = parse_bounding_box(boxes[0][j])
                bboxA=(bbox[0],bbox[1])
                bboxB=(bbox[2],bbox[3])
                frame = cv2.rectangle(frame, bboxA, bboxB, (255,0,0), 2, 1)
                cv2.putText(frame, str(int(classes[0][j])), (bbox[0],bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)

                if User_input_object==int(classes[0][j]):
                    # Define an initial bounding box
                    a = bbox[2]-bbox[0]
                    b = bbox[3]-bbox[1]
                    bbox1 = (bbox[0], bbox[1], a, b)

        cv2.putText(frame, "Detecting", (3,25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)
        # cv2.imshow("img", img) #what see the neural network
        cv2.imshow("Tracking", frame)
        return bbox1,frame

    while True:
        while (bbox==None or ((bbox[0]==0 and bbox[2]==0) and time.time()-start>=2)) and User_input_mode==0:
            bbox,frame = detection(bbox,frame) 
            if (not (bbox==None or (bbox[0]==0 and bbox[2]==0))) and User_input_object!=None:
                tracker = select_tracker(tracker_type) #We need to initialize the Tracker or to reset it if it fails
                # Initialize tracker with first frame and bounding box
                ok = tracker.init(frame, bbox)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit if Q pressed
                break
        
        # Read a new frame
        ok, frame = video.read()
        k= cv2.waitKey(1)

        frame=writelines(frame)

        # time when we finish processing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (3,22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "Step : " + steps[step], (200,22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

        if User_input_object!=None and step<3:
            # Update tracker
            ok, bbox = tracker.update(frame)

            # Display available buttons to press
            cv2.putText(frame, "D: make a detection  F: draw a bounding box manually  Q: exit", (3,475), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0),1)
    
            # Display tracker type on frame
            #cv2.putText(frame, tracker_type + " Tracker", (3,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

    
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                #center = [int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)]
                objbase = [int((p1[0]+p2[0])/2),int(bbox[1] + bbox[3])]
                center = objbase
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                #cv2.circle(frame, (center[0], center[1]), radius=0, color=(255, 0, 0), thickness=10)
                cv2.circle(frame, (objbase[0], objbase[1]), radius=0, color=(255, 0, 0), thickness=10)
                cv2.putText(frame, str(int(finddistance(objbase)))+"cm", (objbase[0], objbase[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),1)
                start = time.time()
    
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (3,65), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display result
        cv2.imshow("Tracking", frame)

        if k & 0xFF == ord('d'): # Press D to make a prediction
            bbox,frame = detection(bbox,frame)
            if not (bbox==None or (bbox[0]==0 and bbox[2]==0)):
                tracker = select_tracker(tracker_type) #We need to initialize the Tracker or to reset it if it fails
                # Initialize tracker with first frame and bounding box
                ok = tracker.init(frame, bbox)

        if k & 0xFF == ord('f'): # Press F to select a Bounding Box manually
            cv2.destroyAllWindows()
            ok, frame = video.read()
            bbox = cv2.selectROI(frame, False)
            tracker = select_tracker(tracker_type) #We need to initialize the Tracker or to reset it if it fails
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(frame, bbox)
            cv2.destroyAllWindows()

        if k & 0xFF == ord('q'): # Exit if Q pressed
            break

y = threading.Thread(target=visual_camera)
y.start()

#f = threading.Thread(target=encoder_L)
#f.start()

#e = threading.Thread(target=encoder_R)
#e.start()

z = threading.Thread(target=comm_alive)
z.start()

#b = threading.Thread(target=batterij_check)
#b.start()


while User_input_mode!=0 and User_input_mode!=1:
    User_input_mode = int(input("\nMain:\n 0 - Automatic Mode\n 1 - Manual Mode\nEnter the desired working mode: "))
    step=1
    if User_input_mode==0:
        print("\nAutomatic Mode starts\n")
    elif User_input_mode==1:
        print("\nManual Mode starts\n")
    elif not User_input_mode==0 and User_input_mode==1:
        print("Error\n")


#Automatic Mode
while User_input_mode==0: #or autoqq_gui==True:

    if User_input_object==None:          
        User_input_object = int(input("Select an object among:\n 0 - Lipton\n 1 - Plastic Cup\n 2 - Sponge \n 3 - Soldering Tin\nEnter the object you would like to pick: "))
        time.sleep(2)   
        step=2               

    #print("STEPeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee: ", step)
    while step == 2: #Walking   #take coordinate object and turn robot or drive straight on

        print("Center of object position: ",center)

        object_centered = False
        object_left = False
        object_right = False
        state = IR_sensor()
        # state = True

        pulsen_L_old = pulsen_list[0]
        #pulsen_R_old = pulsen_list[1]
        theta_old = positie[0]  # in radialen    
        positie_old[0] = positie[0]
        positie_old[1] = positie[1]
        positie_old[2] = positie[2]

    
        if  270 <= center[0] <= 340 and state == True:  #Centred, but object is not yet at the grabber
            rechtdoor(0.2)

            pulsen_L_new = pulsen_list[0] - pulsen_L_old 
            theta_rad = positie[0]   # in radialen opgelet
            afstand_robot = pulsen_L_new * mm_per_puls
            positie[1] = positie_old[1] + afstand_robot * math.cos(theta_rad)
            positie[2] = positie_old[2] + afstand_robot * math.sin(theta_rad)
            hoek_in_graden = (positie[0] * 180) / math.pi
            continue


        elif center[0] < 270:
            links(0.1)

            pulsen_L_new = pulsen_list[0] - pulsen_L_old 
            afstand_robot = pulsen_L_new * mm_per_puls
            theta_new = afstand_robot / (afstand_wielen/2)  # basic formule hoek = afgelegde weg wiel / straal robot
            positie[0] = positie_old[0] + theta_new  # na draaien is dit de nieuwe hoek
            hoek_in_graden = (positie[0] * 180) / math.pi
            continue


        elif center[0] > 340:
            rechts(0.1)

            pulsen_L_new = pulsen_list[0] - pulsen_L_old  
            afstand_robot = pulsen_L_new * mm_per_puls
            theta_new = afstand_robot / (afstand_wielen/2)  # basic formule hoek = afgelegde weg wiel / straal robot
            positie[0] = theta_old - theta_new  # na draaien is dit de nieuwe hoek
            hoek_in_graden = (positie[0] * 180) / math.pi
            continue  

        elif  state == False:  # gecentreerd en voorwerp is aan de grijper
            step = 3


    if step == 3:     # We starten grijpproces om pot_list te voorzien van data voor volgende stap.
        ib = 0
        
        for ib in range(10):
            pot_valuex = potmeter4()  # De waarde van de potentiometer wordt uitgelezen      ==> waarde tussen 0 en 1
            pot_value.append(pot_valuex)
            print(pot_value)
            grijper_sluiten_manueel(i)
            ib = ib + 1
            i = i + 3
            time.sleep(0.1)

        # grijpen tot contact met voorwerp
        treshold = 0.01
        print("dit is de laatste waarde",pot_value[-1],"en dit is de vijfdelaatste",pot_value[-5])
        while (pot_value[-1] + treshold) < pot_value[-4]:  
            #laatste getal in de list   
            
            pot_valuex = potmeter4()  # De waarde van de potentiometer wordt uitgelezen      ==> waarde tussen 0 en 1
            pot_value.append(pot_valuex)
            print(pot_value)
            i = i + 2
            grijper_sluiten_manueel(i)
            time.sleep(0.2)

            x1 = straal * math.cos(i)
            x2 = straal - x1
            y = straal * math.sin(i)

        # grijpen met gewenste kracht
        time.sleep(0.2)

        if User_input_object == 0:
            power = 20
        elif User_input_object == 1:
            power = 10
        elif User_input_object == 2:
            power = 30
        elif User_input_object == 3:
            power = 40

        print("dit is de huidige hoek",i)
        hoek_Rad = (math.pi * i)/ 180
        hoek_graden = ((power/1560) + math.sqrt(0.041225-0.014*math.cos(hoek_Rad)) ** 2 - 0.041225) / (-0.014)
        print("dit is de hoek in graden", hoek_graden)
        i2_rad = math.acos((((power/1560) + math.sqrt(0.041225-0.014*math.cos(hoek_Rad))) ** 2 - 0.041225) / (-0.014))    #met i = de vorige servohoek van de grijper
        print("This is the force: ",i2_rad)
        i2 = (i2_rad * 180) / math.pi
        print("This is the new angle for the power: ",i2)
        grijper_sluiten_manueel(i2)
        time.sleep(0.2)

        step = 4


    elif step == 4:  #Raise arm __> Part of wrist not done yet
        hoek_arm = 0
        while hoek_arm < 30:
            hoek_arm = hoek_arm + 2
            arm_op_en_neer(hoek_arm)
        time.sleep(2)

        step = 5


    elif step == 5:     # Draaien om terug te keren  wijzerszin draaien

        theta_6 = positie[0]
        print("positie: ",positie[0])
        print("theta_6: ",theta_6)
        if theta_6 > 0:     # We draaien terug wijzerszin
            draaiafstand_6 = (afstand_wielen/2) * theta_6
            nodige_pulsen = draaiafstand_6 / mm_per_puls
            pulsen_links = pulsen_list[0]
            pulsen_rechts = pulsen_list[1]
            nodige_pulsen_links = pulsen_links + nodige_pulsen

            while pulsen_list[0] < nodige_pulsen_links:  # zolang we de nodige pulsen niet hebben bereikt blijven we draaien.
                motoren_draaien_Rechts(0.2)
                time.sleep(0.5)
            
            step = 6      # Wanneer de gewenste positie bereikt werd gaan we verder naar stap 7

        elif theta_6 < 0:       # We draaien terug tegenwijzerszin
            theta_6_abs = abs(theta_6)
            draaiafstand_6 = (afstand_wielen/2) * theta_6_abs
            nodige_pulsen = draaiafstand_6 / mm_per_puls
            pulsen_links = pulsen_list[0]
            #pulsen_rechts = pulsen_list[1]
            nodige_pulsen_links = pulsen_links + nodige_pulsen

            while pulsen_list[0] < nodige_pulsen_links:  # zolang we de nodige pulsen niet hebben bereikt blijven we draaien.
                motoren_draaien_Links(0.2)
                time.sleep(0.5)
            
            step = 6      # Wanneer de gewenste positie bereikt werd gaan we verder naar stap 7

        
    elif step == 6: # terug volledige X of Y afstand
        x_coordinaat = positie[1]
        pulsen_links = pulsen_list[0]
        nodige_pulsen = x_coordinaat / mm_per_puls
        nodige_pulsen_totaal = pulsen_links + nodige_pulsen

        while pulsen_list[0] < nodige_pulsen_totaal:  # We gaan terug
            motoren_achter(0.5)
            time.sleep(0.5)
        motoren_stoppen()

        step = 7


    elif step == 7: # draaien om terug volledige X afstand MAAR iets verder
        y_coordinaat = positie[2]
        theta_6 = positie[0]

        if y_coordinaat < 0:
            theta_neg_90 = math.pi/2   #zoveel graden moeten we draaien om evenwijdig te zijn met y-as richting oorsprong
            draaiafstand_6 = (afstand_wielen/2) * theta_neg_90
            nodige_pulsen = draaiafstand_6 / mm_per_puls
            pulsen_links = pulsen_list[0]
            pulsen_rechts = pulsen_list[1]
            nodige_pulsen_links = pulsen_links + nodige_pulsen

            while pulsen_list[0] < nodige_pulsen_links:  # zolang we de nodige pulsen niet hebben bereikt blijven we draaien.
                motoren_draaien_Links(0.2)
                time.sleep(1)
            
            step = 8

        elif y_coordinaat > 0:
            theta_neg_90 = math.pi/2   #zoveel graden moeten we draaien om evenwijdig te zijn met y-as richting oorsprong
            draaiafstand_6 = (afstand_wielen/2) * theta_neg_90
            nodige_pulsen = draaiafstand_6 / mm_per_puls
            pulsen_links = pulsen_list[0]
            pulsen_rechts = pulsen_list[1]
            nodige_pulsen_rechts = pulsen_links + nodige_pulsen

            while pulsen_list[0] < nodige_pulsen_rechts:  # zolang we de nodige pulsen niet hebben bereikt blijven we draaien.
                motoren_draaien_Rechts(0.2)
                time.sleep(1)
            
            step = 8   


    elif step == 8: # terug volledige X of Y afstand
        y_coordinaat = positie[2]
        abs_y_coordinaat = abs(y_coordinaat)
        pulsen_links = pulsen_list[0]
        nodige_pulsen = abs_y_coordinaat / mm_per_puls
        nodige_pulsen_totaal = nodige_pulsen + pulsen_links

        while pulsen_list[0] < nodige_pulsen_totaal:  # We gaan terug
            motoren_rechtdoor(0.2)
            time.sleep(0.5)
        motoren_stoppen()

        step = 9


    elif step == 9: #Assume initial position to pick next object
        hoek_arm1 = 30
        while hoek_arm1 > 0:
            hoek_arm1 = hoek_arm1 - 2
            arm_op_en_neer(hoek_arm1)
            time.sleep(0.2)
        time.sleep(1)
        grijper_sluiten_manueel(0)
        print("\nWork completed")
        time.sleep(5)
        User_input_object=None


#Manual Mode
while User_input_mode==1: #or manual_gui==True   
    # INIT
    i = 0   # ==> de hoogte van de arm begint steeds op hoogte 0.
    a = 0   # ==> de hoek van de grijper
    b = 50   # ==> de hoek van de pols
    wrist_manual(b)
    #comm_alive()        # Led knippert continu om aan te geven dat de robot werkt


    while True:

        #comm_alive()        # Led knippert continu om aan te geven dat de robot werkt
        
        while keyboard.is_pressed('z'): #or Forward_gui==True                             # Naar voor rijde nmanueel
            print("Z-toets werd ingedrukt")
            motoren_rechtdoor(0.1)

        while keyboard.is_pressed('d'):                             # Naar achter rijden manueel
            print("d-toets werd ingedrukt")
            motoren_draaien_Rechts(0.1)

        while keyboard.is_pressed('q'):                             # Draaien naar links
            print("q-toets werd ingedrukt")
            motoren_draaien_Links(0.1)

        while keyboard.is_pressed('s'): #or Backwards_gui==True                         # Draaien naar rechts
            print("s-toets werd ingedrukt")
            motoren_achter(0.1)

        while keyboard.is_pressed('up') or keyboard.is_pressed('down'):         # Arm op en neer sturen manueel
        
            if keyboard.is_pressed('up'):
                i = i + 5
                if i > 180:
                    i = 180
                arm_op_en_neer(i)       # Indien de knop up is ingedrukt zal de hoogte steeds met 1 verhogen.
                
                print('up-key is pressed!!!!')
                print('De hoek is',i)
                
                time.sleep(0.1)     

            elif keyboard.is_pressed('down'):
                i = i - 5
                if i < 0:
                    i = 0
                arm_op_en_neer(i)      # Indien de knop down is ingedrukt zal de hoogte steeds met 1 verlagen.
                
                print('down-key is pressed!!!!')
                print('De hoek is',i)
                time.sleep(0.1)

        
        while keyboard.is_pressed('right') or keyboard.is_pressed('left'):         #Grijper openen en sluiten manueel

            if keyboard.is_pressed('right'):
                a = a + 5
                if a > 180:
                    a = 180
                grijper_sluiten_manueel(a)       # Indien de knop up is ingedrukt zal de hoogte steeds met 1 verhogen.
                
                print('right-key is pressed!!!!')
                print('De hoek is',a)
                
                time.sleep(0.1)     

            elif keyboard.is_pressed('left'):
                a = a - 5
                if a < 0:
                    a = 0

                grijper_sluiten_manueel(a)      # Indien de knop down is ingedrukt zal de hoogte steeds met 1 verlagen.
                
                print('left-key is pressed!!!!')
                print('De hoek is',a)
                time.sleep(0.1)


        while keyboard.is_pressed('p') or keyboard.is_pressed('m'):         # pols bewegen

            if keyboard.is_pressed('p'):
                b = b + 5
                if b > 180:
                    b = 180

                wrist_manual(b)       # Indien de knop up is ingedrukt zal de hoogte steeds met 1 verhogen.
                
                print('p-key is pressed!!!!')
                print('De hoek is',b)
                
                time.sleep(0.1)     

            elif keyboard.is_pressed('m'):
                b = b - 5
                if b < 0:
                    b = 0
                    
                wrist_manual(b)      # Indien de knop down is ingedrukt zal de hoogte steeds met 1 verlagen.
                
                print('m-key is pressed!!!!')
                print('De hoek is',b)
                time.sleep(0.1)


        if  keyboard.is_pressed('enter'):           # We gaan terug naar MAIN.py
            print("You've exit manual mode")
            break
            
        
    #Manual1.manual123()
