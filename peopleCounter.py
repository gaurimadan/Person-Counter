from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap=cv2.VideoCapture("D:\deeplearning\practise\images\people.mp4")
# cap.set(3, 950)
# cap.set(4, 480)

model=YOLO('D:\deeplearning\practise\images\yolov8n.pt')
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
mask=cv2.imread("D:\deeplearning\practise\images\mask-people.png")
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limitsup=[101,161,296,161]
limitsdown=[527,489,735,489]
totalcountup=[]
totalcountdown=[]

while True:
    success, img=cap.read()
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion=cv2.bitwise_and(img, mask_resized)
    results=model(imgRegion,stream=True)

    detections = np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
           
             #cvzone.cornerRect(img,bbox)
            conf=math.ceil((box.conf[0])*100)/100
            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if currentClass=='person'and conf> 0.3:
            
           
                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(30,y1)),
                                   scale=0.8,thickness=1)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack([detections,currentArray])

            
    resultsTracker=tracker.update(detections)
    cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255),5)
    cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)

        print(result)
        w, h = x2-x1, y2-y1
        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsup[0]<cx<limitsup[2] and limitsup[1]-30<cy<limitsup[1]+30:
            if totalcountup.count(id)==0:
             totalcountup.append(id)


        if limitsdown[0]<cx<limitsdown[2] and limitsdown[1]-30<cy<limitsdown[1]+30:
            if totalcountdown.count(id)==0:
             totalcountdown.append(id)
    
    cvzone.putTextRect(img,f'Count Down:{len(totalcountdown)}',(50,120) )
        
    cvzone.putTextRect(img,f'Count Up:{len(totalcountup)}',(50,50) )
        


    cv2.imshow('image',img)
    # cv2.imshow('ImageRegion',imgRegion)
    cv2.waitKey(1)


