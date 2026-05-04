import cv2
import numpy as np
import os

image = cv2.imread("cards.jpg")

if image is None:
    print("画像が見つかりません。cards.jpg を同じフォルダに入れてください。")
    exit()

image = cv2.resize(image, (800, 600))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#出力フォルダ作成
output_dir = "detected_cards"
os.makedirs(output_dir, exist_ok=True)

result = image.copy()
card_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    #ノイズを除外
    if area < 3000:
        continue

    #輪郭を四角形に近似
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    #四角形っぽいものだけカードとして扱う
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)

        #細すぎる・小さすぎるものを除外
        if w < 40 or h < 40:
            continue

        card_count += 1

        #カード部分を切り抜き
        card_img = image[y:y+h, x:x+w]

        #保存
        filename = f"{output_dir}/card_{card_count}.jpg"
        cv2.imwrite(filename, card_img)

        #元画像に枠と番号を描画
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, str(card_count), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

print(f"{card_count}枚のカード画像を保存しました。")

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
