# 프로젝트 개요
- OpenCV 라인 트레이싱
- 웹캠을 이용해 A4 용지 위의 검은색 라인의 중심 좌표를 실시간으로 추적
- OpenCV를 활용해 ROI(관심 영역) 추출 -> 이진화(Thresholding) -> 컨투어 검출 -> 중심점 계산 과정을 거침
- Matplotlib을 이용해 이진화 결과와 히스토그램도 실시간으로 시각화함

---

# 사용한 기술
- Python 3
- OpenCV
- Matplotlib

---

# 구현 과정
## 1. 웹캠 연결 및 해상도 설정
- 웹캠을 열고 해상도를 640x480으로 설정함
```python
# 웹캠 연결
cap = cv2.VideoCapture(0)   # 0번 카메라 장치 연결

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height
```

## 2. 색공간 변환
- 그레이 스케일로 변환함
```python
#색공간 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 결과 출력
cv2.imshow('Original(BGR)', img)
cv2.imshow('HSV', hsv_img)
cv2.imshow('Gray', gray)
```

## 3. 히스토그램 + 오츠 임계값 시각화
- 4가지 이진화 결과(Gray, Threshold 130, Otsu, Adaptive) 표시
```python
# ROI에서 라인만 이진화 (ROI는 다음 단계에서 지정한 후 적용함)
t, t_otsu = cv2.threshold(roi, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
_, t_130 = cv2.threshold(roi, 130, 255, cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                         cv2.THRESH_BINARY, 11, 2)

# 두 개의 figure 생성
plt.ion()   # matplotlib 인터랙티브 모드 / 실시간 영상 처리 + matplotlib 그래프 업데이트를 동시에 처리하기 위한 설정
fig_hist, ax1 = plt.subplots(figsize=(6,4))
fig_otsu, ax2 = plt.subplots(1, 4, figsize=(12,4))

# 이진화 matplotlib 창 갱신
titles = ['ROI Gray', 'Threshold 130', 'Otsu', 'Adaptive']
images = [roi, t_130, t_otsu, adaptive]

for ax, title, im in zip(ax2, titles, images):
    ax.clear()
    ax.set_title(title)
    ax.imshow(im, cmap='gray')
    ax.axis('off')
fig_otsu.suptitle("Thresholding Methods Results (ROI Only)")
fig_otsu.canvas.draw()
plt.pause(0.01)

# 히스토그램
hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
ax1.clear()
ax1.plot(hist)
# 빨간색 점선으로 오츠 알고리즘 임계값을 보여줌
ax1.axvline(x=t, color='r', linestyle="--", label="Otsu t={int(t)}")
ax1.set_title("Histogram (ROI)")
ax1.legend()
fig_hist.canvas.draw()
plt.pause(0.01)
```

## 4. ROI 설정
- 흰색을 기준으로 thresholding을 하고, 가장 큰 컨투어를 A4 용지로 가정
```python
 _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x, y, w, h = 0, 0, gray.shape[1], gray.shape[0]
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 너무 작은 영역이면 무시 (A4용지가 아닐 가능성)
    if w*h > 5000:
      roi = gray[y:y+h, x:x+w]
    else:
      roi = gray
      x, y = 0, 0
else:
    roi = gray
    x, y = 0, 0
```
## 5. ROI에서 라인만 이진화
- 오츠 알고리즘을 이요해 ROI 내부의 라인을 자동으로 이진화
- t 값은 계산된 임계값
```python
t, t_otsu = cv2.threshold(roi, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

## 6. 라인 컨투어 검출 및 중심 좌표 계산
```python
line_mask = cv2.bitwise_not(t_otsu)
line_contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if line_contours:
  # 가장 큰 컨투어를 라인으로 가정
  line_c = max(line_contours, key=cv2.contourArea)
  M = cv2.moments(line_c)

  if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    global_cx = x + cx
    global_cy = y + cy
    print(f"라인 중심 좌표 : ({global_cx}, {global_cy})")

    # 전체 화면에 컨투어와 중심점 표시
    gray_center = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(gray_center[y:y+h, x:x+w], [line_c], -1, (0,255,0),2)
    cv2.circle(gray_center, (global_cx, global_cy), 5, (0, 0, 255), -1)
    cv2.imshow("Center", gray_center)
```

---

# 시행착오
- 1. 중심점이 안 움직임(수정 O)
  2. 흰색 배경까지 contour로 잡힘(수정 X)
  3. A4 바깥 contour가 잡힘(수정 X)
