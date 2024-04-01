import cv2

cap = cv2.VideoCapture('/home/uavlab20/Downloads/car.mp4')

ret = True
f = 0 
while ret:
    print(f'frame: {f}')    
    ret, img = cap.read()
    f+=1