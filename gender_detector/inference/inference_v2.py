import argparse
import cv2
import os
import uuid
import utils

def capture_image_from_camera(model, scaler):
    """Open webcam and take a snapshot when the user presses SPACE."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return None

    print("Press SPACE to take a photo, ESC to quit.")
    img_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Press SPACE to capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:  # SPACE
            filename = f"snapshot_{uuid.uuid4().hex[:8]}.jpg"
            img_path = os.path.join("images", filename)
            cv2.imwrite(img_path, frame)
            print(f"Saved snapshot to: {img_path}")
            utils.predict(model, scaler, img_path)

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model/logreg_gender_model.joblib")
    parser.add_argument('--scaler', type=str, default="model/logreg_scaler.joblib")
    args = parser.parse_args()

    model, scaler = utils.load_model(args.model, args.scaler)

    capture_image_from_camera(model, scaler)

if __name__ == "__main__":
    main()
