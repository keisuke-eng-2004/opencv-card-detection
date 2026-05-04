import cv2
import numpy as np
import os

image = cv2.imread("cards.jpg")

if image is None:
    print("画像が見つかりません。cards.jpg を同じフォルダに入れてください。")
    exit()

image = cv2.resize(image, (800, 600))

# 出力フォルダ
output_dir = "detected_cards"
os.makedirs(output_dir, exist_ok=True)

# 前処理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 白いカード部分を抽出
_, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

# ノイズ除去
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 輪郭検出
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = image.copy()
card_count = 0

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下

    return rect

for cnt in contours:
    area = cv2.contourArea(cnt)

    # 小さいノイズ除去
    if area < 3000:
        continue

    # 輪郭を四角形に近似
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

    if len(approx) == 4:
        rect = order_points(approx)

        # 出力サイズ
        width = 250
        height = 350

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # 透視変換
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        card_count += 1
        filename = f"{output_dir}/card_{card_count}.jpg"
        cv2.imwrite(filename, warped)

        # 元画像に枠と番号
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(result, str(card_count), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

print(f"{card_count}枚のカードを補正して保存しました。")

cv2.imshow("Threshold", thresh)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
