import cv2
import numpy as np

#画像ファイル読込み
image = cv2.imread("sample.jpg")

if image is None:
    print("画像が見つかりません。sample.jpg を同じフォルダに入れてください。")
    exit()

#サイズを調整
image = cv2.resize(image, (800, 600))

#グレースケール
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#ぼかし
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#エッジ検出
edges = cv2.Canny(blur, 50, 150)

#輪郭検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#ノイズを除外
filtered_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:
        filtered_contours.append(cnt)

#輪郭を描画
result = image.copy()
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

#結果
cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edges", edges)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
