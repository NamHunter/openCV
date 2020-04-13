import cv2

cap = cv2.VideoCapture('video/slow_traffic_small.mp4')

tracker = cv2.TrackerCSRT_create()
_,frame = cap.read()
bbox= cv2.selectROI("Tracking",frame,False)
tracker.init(frame,bbox)


def drawBox(frame,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(frame,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(frame,"Tracking",(100,125),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    return
while True:
    timer = cv2.getTickCount()
    ret,frame = cap.read()

    ret,bbox = tracker.update(frame)

    if ret:
        drawBox(frame,bbox)
    else:
        cv2.putText(frame,"Lost",(100,125),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    print(fps)
    cv2.putText(frame,str(int(fps)),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow("Tracking",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows