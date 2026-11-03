  # save this as test_webcam.py and run it
import cv2
cap = cv2.VideoCapture(0)   # 0 means first webcam
ret, frame = cap.read()
if ret:
    print('Webcam working!')
    cv2.imwrite('test_frame.jpg', frame)
else:
    print('Webcam NOT found. Check connection.')
cap.release()
