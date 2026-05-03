import cv2
import numpy as np

#画像ファイルを読み込む
image = cv2.imread("sample.jpg")

if image is None:
    print("画像が見つかりません。sample.jpg を同じフォルダに入れてください。")
    exit()

#画像サイズを調整
image = cv2.resize(image, (800, 600))

#グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#ぼかし処理
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#エッジ検出
edges = cv2.Canny(blur, 50, 150)

#輪郭検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#輪郭を描画
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

#結果を表示
cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edges", edges)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
