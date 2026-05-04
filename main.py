import cv2
import numpy as np
import os

# 画像読み込み
image = cv2.imread("sample.jpg")

if image is None:
    print("sample.jpg が見つかりません")
    exit()

# サイズ調整
image = cv2.resize(image, (800, 600))
result = image.copy()

# 出力フォルダ
os.makedirs("output", exist_ok=True)

# グレースケール
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ぼかし
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# エッジ検出
edges = cv2.Canny(blur, 40, 120)

# 輪郭をつなげる
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 輪郭検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

card_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    # 面積でノイズ除去
    if area < 3000:
        continue

    # 回転した四角形で囲む
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    w, h = rect[1]

    if w == 0 or h == 0:
        continue

    ratio = max(w, h) / min(w, h)

    # トランプっぽい縦横比だけ残す
    if ratio < 1.2 or ratio > 2.5:
        continue

    card_count += 1

    # 枠を描画
    cv2.drawContours(result, [box], 0, (0, 255, 0), 2)

    x, y, bw, bh = cv2.boundingRect(box)
    cv2.putText(result, str(card_count), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

print(f"{card_count}枚のカード候補を検出しました")

# 結果を保存
cv2.imwrite("output/result.jpg", result)
cv2.imwrite("output/edges.jpg", edges)

# 表示
cv2.imshow("Edges", edges)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
