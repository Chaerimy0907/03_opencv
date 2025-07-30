import cv2
import numpy as np
import matplotlib.pylab as plt

# 웹캠 연결
cap = cv2.VideoCapture(0)   # 0번 카메라 장치 연결

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

# 두 개의 figure 생성
plt.ion()   # matplotlib 인터랙티브 모드 / 실시간 영상 처리 + matplotlib 그래프 업데이트를 동시에 처리하기 위한 설정
fig_hist, ax1 = plt.subplots(figsize=(6,4))
fig_otsu, ax2 = plt.subplots(1, 3, figsize=(10,4))

if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()           # 다음 프레임 읽기
        
        if ret:
            # q 입력 받으면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
#                cv2.imwrite('../img/line.jpg', gray_img)   
                break                   
        else:
            print('no frame')
            break
        
        # 색공간 변환
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 결과 출력
        cv2.imshow('Original(BGR)', img)
        #cv2.imshow('HSV', hsv_img)
        cv2.imshow('Gray', gray_img)

        # 히스토그램
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        ax1.clear()
        ax1.plot(hist)
        
        # 빨간색 점선으로 오츠 알고리즘 임계값을 보여줌
        ax1.axvline(x=t, color='r', linestyle="--", label=f"Otsu t={int(t)}")
        ax1.set_title("Histogram")
        ax1.legend()

        fig_hist.canvas.draw()
        plt.pause(0.01)

        # 오츠 알고리즘
        # 경계 값을 130으로 지정
        _, t_130 = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)

        # 경계 값을 지정하지 않고 OTSU 알고리즘 선택
        t, t_otsu = cv2.threshold(gray_img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print('otsu threshold : ', t)

        # matplotlib 창 갱신
        titles = ['Original', 'Threshold 130', f'Otsu {int(t)}']
        images = [gray_img, t_130, t_otsu]

        for ax, title, im in zip(ax2, titles, images):
            ax.clear()
            ax.set_title(title)
            ax.imshow(im, cmap='gray')
            ax.axis('off')
        fig_otsu.suptitle("Otsu's Method Results")
        fig_otsu.canvas.draw()
        plt.pause(0.01)
        
else:
    print("can't open camera.")

cap.release()                           # 자원 반납
cv2.destroyAllWindows()