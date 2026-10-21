import os
import sys

import cv2

NAME = "Catalina"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(BASE_DIR, "test", "i1.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUT_PATH = os.path.join(OUTPUT_DIR, "env_test_result.jpg")

def main():
    
    img = cv2.imread(IMG_PATH)
    if img is None:
        sys.exit(f"Could not read image: {IMG_PATH}")
    print(f"Loaded {IMG_PATH}, shape (h, w, channels) = {img.shape}")

    cv2.imshow("Original", img)
    cv2.waitKey(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.5, 3
    (text_w, text_h), baseline = cv2.getTextSize(NAME, font, scale, thickness)
    h, w = img.shape[:2]
    org = ((w - text_w) // 2, (h + text_h) // 2)

    cv2.putText(img, NAME, org, font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(img, NAME, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("With name", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if cv2.imwrite(OUT_PATH, img):
        print(f"Saved {OUT_PATH}")
    else:
        sys.exit(f"Could not save image: {OUT_PATH}")


if __name__ == "__main__":
    main()
