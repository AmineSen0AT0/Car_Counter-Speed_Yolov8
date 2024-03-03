from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import yaml
import math
import time
from sort import *
from PIL import Image, ImageEnhance, ImageFilter
#
#### enhacing video frames ########
def apply_pil_enhancements(frame):
    pil_image = Image.fromarray(frame)  # Convert OpenCV frame to PIL image

    # Example enhancement: Adjusting brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced_image = enhancer.enhance(2.5)  # Increase brightness by a factor of 2.5

    # Apply sharpen filter to the enhanced image
    sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)

    # Convert the sharpened PIL image back to an OpenCV frame
    sharpened_frame = np.array(sharpened_image)
    return sharpened_frame


capture = cv2.VideoCapture("Videos/TEST V1.mp4") # for video
#capture = cv2.VideoCapture(0) # for webcam
capture.set(3 , 1280)
capture.set(4 , 720)

fps = 0
distance = 50
speedInKm = 0
totalVhiclesCounter = set()
enterDictionary = {}
speedDictionary = {}


model = YOLO('yolov8n.pt')
model.export(format='openvino')
ov_model=YOLO('yolov8n_openvino_model')

# Tracking
tracker = Sort(max_age=20, min_hits= 3, iou_threshold= 0.3)

# those are lines points to draw theme
limitsL1 = [350, 50, 510, 50]
limitsL2 = [55, 200, 470, 200]

limitsL3 = [740, 350, 1250, 350]
limitsL4 = [650, 150, 1000, 150]

coco_yaml_path = 'coco8.yaml'

# Load the COCO dataset YAML file
# with open(coco_yaml_path, 'r') as file:
#     coco_data = yaml.safe_load(file)

# Extract class names from COCO dataset
# classNames = coco_yaml_path[4]#['names']

# ClassNames from coco dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    timeStart = time.time()
    success , img= capture.read()

    results = ov_model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes= r.boxes
        for box in boxes:
            x1,y1,x2,y2= box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2) #this for opencv
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img, (x1,y1),(x2,y2) , (0,200, 0), 3)
            w, h =x2-x1 , y2-y1
            # cvzone.cornerRect(img, (x1,y1,w,h) , l=10)
            # Confidence
            confidence = math.floor(box.conf[0]*100)/100
            # print(confidence)
            # Class Name
            detectionClass = int(box.cls[0])
            currentClass=classNames[detectionClass]
            if currentClass== "car" or currentClass=="truck" or currentClass=="bus" or currentClass=="motorbike" and confidence  >0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=10 , rt=5)
                cvzone.putTextRect(img , f'{classNames[detectionClass]} {confidence}' , (max(0,x1),max(35,y1)) , scale=1 , thickness=1)
                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img,(limitsL1[0], limitsL1[1]), (limitsL1[2], limitsL1[3]), (255,0,0), 5)
    cv2.line(img,(limitsL2[0], limitsL2[1]), (limitsL2[2], limitsL2[3]), (0,0,255), 5)
    cv2.line(img,(limitsL3[0], limitsL3[1]), (limitsL3[2], limitsL3[3]), (255,0,0), 5)
    cv2.line(img,(limitsL4[0], limitsL4[1]), (limitsL4[2], limitsL4[3]), (0,0,255), 5)


    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # this for opencv
        w, h = x2 - x1, y2 - y1
        print(result)
        #cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255,0,0))
        #cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        centerX, centerY = x1 +w//2 , y1+h//2
        cv2.circle(img,(centerX,centerY), 5, (255,0,255), cv2.FILLED)

        if limitsL1[0]< centerX < limitsL1[2] and limitsL1[1]-20 < centerY < limitsL1[1] + 20:
            totalVhiclesCounter.add(id)
            enterDictionary[id]= time.time()
            cv2.line(img, (limitsL1[0], limitsL1[1]), (limitsL1[2], limitsL1[3]), (0, 255, 0), 5)
        if id in enterDictionary:
            if limitsL2[0]< centerX < limitsL2[2] and limitsL2[1]-20 < centerY < limitsL2[1] + 20:
                cv2.line(img, (limitsL2[0], limitsL2[1]), (limitsL2[2], limitsL2[3]), (0, 255, 0), 5)
                arrivalTime= time.time() - enterDictionary[id]
                speedTime= distance/arrivalTime
                speedInKm = speedTime * 3.6
                speedDictionary[id] = speedInKm

                # cv2.putText(img, str(int(speedInKm)+ 'km/h'), )
            if id in speedDictionary:
                cvzone.putTextRect(img, f'{int(speedDictionary[id])} Km/h', (max(0, x2), max(35, y2)), scale=1, thickness=1)
                if speedDictionary[id] > 50:
                    cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5, colorR=(0, 0, 255))
                    # cvzone.putTextRect(img, f'{classNames[detectionClass]} {confidence}', (max(0, x1), max(35, y1)),scale=1, thickness=1)

        if limitsL3[0] < centerX < limitsL3[2] and limitsL3[1] - 20 < centerY < limitsL3[1] + 20:
            totalVhiclesCounter.add(id)
            enterDictionary[id] = time.time()
            cv2.line(img, (limitsL3[0], limitsL3[1]), (limitsL3[2], limitsL3[3]), (0, 255, 0), 5)
        if id in enterDictionary:
            if limitsL4[0] < centerX < limitsL4[2] and limitsL4[1] - 20 < centerY < limitsL4[1] + 20:
                cv2.line(img, (limitsL4[0], limitsL4[1]), (limitsL4[2], limitsL4[3]), (0, 255, 0), 5)
                arrivalTime = time.time() - enterDictionary[id]
                speedTime = distance / arrivalTime
                speedInKm = speedTime * 3.6
                speedDictionary[id] = speedInKm
            if id in speedDictionary:
                cvzone.putTextRect(img, f'{int(speedDictionary[id])} Km/h', (max(0, x2), max(35, y2)), scale=1,
                                   thickness=1)
                if speedDictionary[id] > 50:
                    cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5, colorR=(0, 0, 255))
        with open('speed.txt', 'w') as file:
            # Write text with string formatting
            file.write("SPEED Information:\n")
            for key, value in speedDictionary.items():
                file.write(f"id : {key}: speed in kmh: {value}\n")

    cvzone.putTextRect(img, f'Count: {len(totalVhiclesCounter)}', (1000, 60))
    # cv2.putText(img,str(len(totalVhiclesCounter)),(230,90), cv2.FONT_HERSHEY_PLAIN, 5, (100,100,100), 5)
    timeEnd = time.time()
    loopTime = timeEnd - timeStart
    fps = fps * .9 + .1 * (1 / loopTime)
    cv2.putText(img, str(int(fps)) + ' FPS', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2. imshow("Image" , img)
    cv2.waitKey(1)
