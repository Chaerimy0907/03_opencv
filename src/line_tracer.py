import cv2
import numpy as np
import matplotlib.pylab as plt

# 웹캠 연결
cap = cv2.VideoCapture(0)               # 0번 카메라 장치 연결 ---①

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height


if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()           # 다음 프레임 읽기

         # 색공간 변환
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 결과 출력
        cv2.imshow('Original(BGR)', img)
        cv2.imshow('HSV', hsv_img)
        cv2.imshow('Gray', gray_img)

        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        plt.plot(hist)

        print("hist.shape : ", hist.shape)
        print("hist.sum() : ", hist.sum(), "img:shape : ",gray_img.shape)
        plt.show()

        if ret:
            # q 입력 받으면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                cv2.imwrite('../img/line.jpg', gray_img)   
                break                   
        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()                           # 자원 반납
cv2.destroyAllWindows()