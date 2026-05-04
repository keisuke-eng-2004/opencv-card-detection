import cv2
import numpy as np
import os

image = cv2.imread("sample.jpg")

if image is None:
    print("画像が見つかりません。sample.jpg を同じフォルダに入れてください。")
    exit()

image = cv2.resize(image, (800, 600))

output_dir = "detected_cards"
os.makedirs(output_dir, exist_ok=True)

# 古い出力画像を消す
for file in os.listdir(output_dir):
    if file.endswith(".jpg"):
        os.remove(os.path.join(output_dir, file))

result = image.copy()

# -----------------------------
# 1. 前処理：明るさ補正
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# コントラスト補正
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

# ぼかし
blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

# -----------------------------
# 2. 白いカード領域を抽出
# -----------------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 白っぽい部分を抽出
lower_white = np.array([0, 0, 120])
upper_white = np.array([180, 80, 255])
white_mask = cv2.inRange(hsv, lower_white, upper_white)

# -----------------------------
# 3. エッジ検出も併用
# -----------------------------
edges = cv2.Canny(blur, 30, 100)

# 白マスクとエッジを合成
combined = cv2.bitwise_or(white_mask, edges)

# ノイズ除去・輪郭をつなぐ
kernel = np.ones((5, 5), np.uint8)
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
combined = cv2.dilate(combined, kernel, iterations=1)

# -----------------------------
# 4. 輪郭検出
# -----------------------------
contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

card_count = 0

def order_points(pts):
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

    # 面積でノイズ除去
    if area < 2500 or area > 40000:
        continue

    # 回転した四角形として検出
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    w, h = rect[1]

    if w == 0 or h == 0:
        continue

    # 縦横比チェック
    ratio = max(w, h) / min(w, h)

    if ratio < 1.2 or ratio > 2.2:
        continue

    # 小さすぎるもの除外
    if max(w, h) < 80:
        continue

    ordered = order_points(box)

    output_width = 250
    output_height = 350

    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (output_width, output_height))

    card_count += 1
    cv2.imwrite(f"{output_dir}/card_{card_count}.jpg", warped)

    # 結果表示用
    box_int = box.astype(int)
    cv2.drawContours(result, [box_int], 0, (0, 255, 0), 2)

    x, y = box_int[0]
    cv2.putText(result, str(card_count), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

print(f"{card_count}枚のカードを保存しました。")

cv2.imshow("White Mask", white_mask)
cv2.imshow("Combined", combined)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
