import cv2
from tracker import *

tracker = Tracker()

cap = cv2.VideoCapture("clip.mp4")
 
 # khử nền
obj_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=35) 

cnt_obj = 0
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    i += 1
    if i %  5 != 0:
        continue
    frame=cv2.resize(frame,(640,352))
    height, width,_ = frame.shape
    #[200 : 330, 0: 640]
    #88 [110: 185, 250: 640]
    reg = frame[200 : 330, 0: 640]

    mask = obj_detector.apply(reg)
    
    # xử lý phân ngưỡng và khử muối tiêu
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    median  = cv2.medianBlur(mask, 3)
   
   # Lấy vẽ đường viền
    contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    list =[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if  area > 3000:
            #cv2.drawContours(reg, [cnt], -1,(0,255,0),2)
            x, y ,w, h = cv2.boundingRect(cnt)
            list.append([x, y, w, h])
    #tracking        
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x, y, w, h, id = bbox
        cv2.rectangle(reg, (x,y),(x+w, y + h), (0,255,0),3)

       #cv2.putText(reg,str(int(id)),(x,y-15),cv2.FONT_HERSHEY_TRIPLEX,0.5,(255, 0, 0),1)

        cnt_obj = int(id)
    cv2.putText(frame,str(cnt_obj),(70,80 ),cv2.FONT_HERSHEY_TRIPLEX,1,(255, 0, 0),1)

    cv2.imshow("Median" , median)
    cv2.imshow("Region", reg)   
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()