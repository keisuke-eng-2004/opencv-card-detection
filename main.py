import cv2
import numpy as np
import os

image = cv2.imread("cards.jpg")

if image is None:
    print("画像が見つかりません。cards.jpg を同じフォルダに入れてください。")
    exit()

image = cv2.resize(image, (800, 600))

output_dir = "detected_cards"
os.makedirs(output_dir, exist_ok=True)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 明るさのムラを減らす
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# エッジ検出
edges = cv2.Canny(blur, 30, 100)

# 線を少し太くして輪郭をつなげる
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = image.copy()
card_count = 0

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

for cnt in contours:
    area = cv2.contourArea(cnt)

    # 小さいノイズ除去
    if area < 2500:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)

        # サイズ条件
        if w < 50 or h < 50:
            continue

        # 縦横比チェック
        ratio = max(w, h) / min(w, h)
        if ratio < 1.2 or ratio > 2.2:
            continue

        rect = order_points(approx)

        width = 250
        height = 350

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        card_count += 1
        cv2.imwrite(f"{output_dir}/card_{card_count}.jpg", warped)

        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
        cv2.putText(result, str(card_count), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

print(f"{card_count}枚のカードを保存しました。")

cv2.imshow("Edges", edges)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
