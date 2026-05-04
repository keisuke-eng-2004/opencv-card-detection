import cv2
import numpy as np
import os

IMAGE_NAME = "cards.jpg"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, file)

    if file.endswith(".jpg") or file.endswith(".png"):
        os.remove(file_path)

print("前回の出力画像を削除しました")

image = cv2.imread(IMAGE_NAME)

if image is None:
    print(f"{IMAGE_NAME} が見つかりません")
    exit()

image = cv2.resize(image, (800, 600))
result = image.copy()

manual_points = []
manual_count = 0

#4点を左上・右上・右下・左下の順に並び替える
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      
    rect[2] = pts[np.argmax(s)]      
    rect[1] = pts[np.argmin(diff)]   
    rect[3] = pts[np.argmax(diff)]   
    return rect


#カードを正面補正
def save_card_from_points(img, pts, filename):
    ordered = np.array(pts, dtype="float32")
    
    width = 250
    height = 350

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, M, (width, height))


    h, w = warped.shape[:2]
    if w > h:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    tl = np.mean(gray[0:50, 0:50])
    tr = np.mean(gray[0:50, -50:])
    bl = np.mean(gray[-50:, 0:50])
    br = np.mean(gray[-50:, -50:])

    values = [tl, tr, br, bl]
    min_idx = np.argmin(values)

    if min_idx == 1:  
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif min_idx == 2:  
        warped = cv2.rotate(warped, cv2.ROTATE_180)
    elif min_idx == 3:  
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite(filename, warped)
    return warped

#自動検出

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

kernel = np.ones((5, 5), np.uint8)
combined = cv2.dilate(edges, kernel, iterations=2)
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

auto_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 2000 or area > 50000:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

    if len(approx) != 4:
        continue

    box = approx.reshape(4, 2).astype("float32")

    x, y, w, h = cv2.boundingRect(box.astype(int))

    if w == 0 or h == 0:
        continue

    ratio = max(w, h) / min(w, h)

    if ratio < 1.0 or ratio > 3.5:
        continue

    auto_count += 1

    save_card_from_points(
        image,
        box,
        f"{OUTPUT_DIR}/auto_card_{auto_count}.jpg"
    )

    cv2.drawContours(result, [box.astype(int)], -1, (255, 0, 0), 3)
    cv2.putText(result, f"A{auto_count}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)



#手動補正用処理
def mouse_callback(event, x, y, flags, param):
    global manual_points, manual_count, result

    if event == cv2.EVENT_LBUTTONDOWN:
        manual_points.append([x, y])

        # 点を表示
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(result, str(len(manual_points)), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4点クリックしたら保存
        if len(manual_points) == 4:
            manual_count += 1

            pts = np.array(manual_points, dtype="float32")

            save_card_from_points(
                image,
                pts,
                f"{OUTPUT_DIR}/manual_card_{manual_count}.jpg"
            )

            cv2.polylines(result, [pts.astype(int)], True, (0, 0, 255), 3)
            cv2.putText(result, f"M{manual_count}", tuple(pts[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            print(f"manual_card_{manual_count}.jpg を保存しました")

            manual_points = []


#表示・操作

cv2.namedWindow("Auto + Manual Detection")
cv2.setMouseCallback("Auto + Manual Detection", mouse_callback)

print("自動検出完了:", auto_count, "枚")
print("未検出のカードは、角を4点クリックしてください")
print("クリック順はだいたいでOKです")
print("終了するには q を押してください")
print("取り消しは r を押してください")

while True:
    cv2.imshow("Auto + Manual Detection", result)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break

    # 直近の手動クリックをリセット
    if key == ord("r"):
        manual_points = []
        result = image.copy()
        print("手動クリックをリセットしました")

cv2.imwrite(f"{OUTPUT_DIR}/final_result.jpg", result)
cv2.imwrite(f"{OUTPUT_DIR}/combined.jpg", combined)

cv2.destroyAllWindows()

print("完了しました")
print(f"自動検出: {auto_count}枚")
print(f"手動補正: {manual_count}枚")
print("output フォルダを確認してください")
