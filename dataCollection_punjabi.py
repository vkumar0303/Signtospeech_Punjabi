import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# ==============================
# CONFIGURATION - PUNJABI DATASET
# ==============================
offset = 20
imgSize = 300
cooldown = 5

# Data collection settings
settings = {
    "min_samples": 50,
    "max_samples": 200,
}

# Punjabi Gurmukhi alphabet gestures (75 gestures)
gestures = [
    "Oora",
    "Airaa",
    "Eeree",
    "Saassa",
    "Haaha",
    "Kakkaa",
    "Khakhhaa",
    "Gaggaa",
    "Ghagghaa",
    "Nganngaa",
    "Chachaa",
    "Chhachhaa",
    "Jajaa",
    "Jhajhaa",
    "Nyaanyaa",
    "Ttainkaa",
    "Tthaithhaa",
    "Ddaidaa",
    "Ddhaiddhaa",
    "Nnaahnaa",
    "Tataa",
    "Thathaa",
    "Dadaa",
    "Dhadhaa",
    "Nanaa",
    "Papa",
    "Phapha",
    "Baba",
    "Bhabha",
    "Mama",
    "Yayaa",
    "Raraa",
    "Lalaa",
    "Vavaa",
    "Rharhaa",
    "Shasha",
    "Khashapha",
    "Zazaa",
    "Fafaa",
    "Khakhha",
    "Gagga",
    "Zazha",
    "Reyra",
    "Layla",
    "Vava",
    "Adhak",
    "Bindi",
    "Tippi",
    "Visarga",
    "Space",
    "Delete",
    "Comma",
    "Period",
    "Question",
]

TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
WARNING_COLOR = (0, 0, 255)
INFO_COLOR = (0, 255, 0)
CONFIRM_COLOR = (0, 255, 0)
FONT_SCALE = 1.2
FONT_THICKNESS = 2

# Hand landmark connections for skeleton drawing
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky finger
    (5, 9),
    (9, 13),
    (13, 17),  # Palm connections
]

# ==============================
# INITIALIZATION
# ==============================
if not os.path.exists("Punjabi_Data"):
    os.makedirs("Punjabi_Data")
    print("Created Punjabi_Data directory")

# Create subdirectories for different data types
for data_type in ["landmarks", "images"]:
    if not os.path.exists(f"Punjabi_Data/{data_type}"):
        os.makedirs(f"Punjabi_Data/{data_type}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device")
    exit(1)

detector = HandDetector(maxHands=1, detectionCon=0.7)

current_gesture_index = 0
folder_landmarks = f"Punjabi_Data/landmarks/{gestures[current_gesture_index]}"
folder_images = f"Punjabi_Data/images/{gestures[current_gesture_index]}"
os.makedirs(folder_landmarks, exist_ok=True)
os.makedirs(folder_images, exist_ok=True)

last_action_time = time.time()
counter = 0
confirmation_time = 0
skip_next = False


# ==============================
# LANDMARK PROCESSING FUNCTIONS
# ==============================
def normalize_landmarks(landmarks, img_width, img_height):
    """Normalize landmarks to 0-1 range"""
    normalized = []
    for lm in landmarks:
        normalized.append([lm[0] / img_width, lm[1] / img_height])
    return normalized


def draw_landmarks_on_white_bg(landmarks, img_size=300):
    """Draw hand landmarks skeleton on white background"""
    # Create white background
    landmark_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    if not landmarks:
        return landmark_img

    # Convert normalized landmarks back to image coordinates
    points = []
    for lm in landmarks:
        x = int(lm[0] * img_size)
        y = int(lm[1] * img_size)
        points.append((x, y))

    # Draw connections (skeleton)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(landmark_img, points[start_idx], points[end_idx], (0, 0, 0), 3)

    # Draw landmark points
    for i, point in enumerate(points):
        # Different colors for different parts of hand
        if i == 0:  # Wrist
            color = (255, 0, 0)  # Red
        elif i in [4, 8, 12, 16, 20]:  # Fingertips
            color = (0, 255, 0)  # Green
        else:  # Other joints
            color = (0, 0, 255)  # Blue

        cv2.circle(landmark_img, point, 6, color, -1)
        cv2.circle(landmark_img, point, 6, (0, 0, 0), 2)  # Black border

    return landmark_img


def process_hand_data(hands, img_shape):
    """Process hand data to extract both image crop and landmarks"""
    if not hands:
        return None, None, None

    hand = hands[0]
    x, y, w, h = hand["bbox"]
    landmarks = hand["lmList"]

    # Get image dimensions
    img_height, img_width = img_shape[:2]

    # Normalize landmarks
    normalized_landmarks = normalize_landmarks(landmarks, img_width, img_height)

    # Create landmark skeleton image
    landmark_img = draw_landmarks_on_white_bg(normalized_landmarks, imgSize)

    # Create bounding box for cropping
    x1 = max(0, x - offset - 15)
    y1 = max(0, y - offset - 15)
    x2 = min(img_width, x + w + offset + 15)
    y2 = min(img_height, y + h + offset + 15)

    return (x1, y1, x2, y2), normalized_landmarks, landmark_img


# ==============================
# UI FUNCTIONS
# ==============================
def draw_stable_ui(img, gesture, count, time_remaining):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 120), BACKGROUND_COLOR, cv2.FILLED)

    cv2.putText(
        img,
        f"Punjabi: {gesture}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )
    cv2.putText(
        img,
        f"Progress: {current_gesture_index+1}/{len(gestures)}",
        (w // 2, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        img,
        f"Images: {count}/{settings['max_samples']}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE - 0.2,
        INFO_COLOR if count >= settings["min_samples"] else WARNING_COLOR,
        FONT_THICKNESS,
    )

    cv2.putText(
        img,
        "Saving: Images + Landmarks",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    timer_text = (
        f"Next: {time_remaining}s" if time_remaining > 0 else "Ready for Next Gesture!"
    )
    text_size = cv2.getTextSize(
        timer_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE - 0.2, FONT_THICKNESS
    )[0]
    cv2.putText(
        img,
        timer_text,
        (w - text_size[0] - 20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE - 0.2,
        WARNING_COLOR if time_remaining > 0 else INFO_COLOR,
        FONT_THICKNESS,
    )

    help_text = "Press 'S' to Capture | 'Q' Quit | 'N' Skip"
    cv2.putText(
        img, help_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1
    )

    if time.time() - confirmation_time < 1.0:
        cv2.putText(
            img,
            "SAVED BOTH!",
            (w // 2 - 120, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            CONFIRM_COLOR,
            3,
        )

    if skip_next:
        cv2.putText(
            img,
            "SKIPPING NEXT GESTURE",
            (w // 2 - 180, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )


# ==============================
# MAIN LOOP
# ==============================
print("Starting Enhanced Punjabi Sign Language Data Collection...")
print(
    f"Collecting {settings['min_samples']}-{settings['max_samples']} samples per gesture"
)
print("Saving both hand images and landmark skeleton data...")

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    hands, img_with_hands = detector.findHands(img, draw=True)

    current_time = time.time()
    elapsed = current_time - last_action_time
    time_remaining = max(0, cooldown - int(elapsed))

    draw_stable_ui(
        img_with_hands, gestures[current_gesture_index], counter, time_remaining
    )

    # Auto-advance logic
    if elapsed >= cooldown and not skip_next:
        if counter >= settings["min_samples"]:
            current_gesture_index += 1
            last_action_time = current_time
            if current_gesture_index >= len(gestures):
                print("\nData collection complete!")
                time.sleep(3)
                break
            folder_landmarks = (
                f"Punjabi_Data/landmarks/{gestures[current_gesture_index]}"
            )
            folder_images = f"Punjabi_Data/images/{gestures[current_gesture_index]}"
            os.makedirs(folder_landmarks, exist_ok=True)
            os.makedirs(folder_images, exist_ok=True)
            counter = 0
        else:
            print(f"WARNING: Only {counter}/{settings['min_samples']} samples!")

    # Skip logic
    if skip_next:
        current_gesture_index += 1
        last_action_time = current_time
        skip_next = False
        if current_gesture_index >= len(gestures):
            print("\nData collection complete after skip!")
            time.sleep(3)
            break
        folder_landmarks = f"Punjabi_Data/landmarks/{gestures[current_gesture_index]}"
        folder_images = f"Punjabi_Data/images/{gestures[current_gesture_index]}"
        os.makedirs(folder_landmarks, exist_ok=True)
        os.makedirs(folder_images, exist_ok=True)
        counter = 0

    # Process hand data
    processed_data = process_hand_data(hands, img.shape)
    imgWhite = None
    landmark_img = None

    if processed_data[0] is not None:
        bbox, normalized_landmarks, landmark_img = processed_data
        x1, y1, x2, y2 = bbox

        # Create traditional processed image
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            h_crop, w_crop = imgCrop.shape[:2]
            aspectRatio = h_crop / w_crop

            try:
                if aspectRatio > 1:
                    k = imgSize / h_crop
                    wCal = math.ceil(k * w_crop)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = (imgSize - wCal) // 2
                    imgWhite[:, wGap : wGap + wCal] = imgResize
                else:
                    k = imgSize / w_crop
                    hCal = math.ceil(k * h_crop)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = (imgSize - hCal) // 2
                    imgWhite[hGap : hGap + hCal, :] = imgResize

                # Show both processed image and landmarks
                cv2.imshow("Processed Hand Image", imgWhite)
                cv2.imshow("Hand Landmarks Skeleton", landmark_img)

            except Exception as e:
                print(f"Processing error: {str(e)}")

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if (
        key == ord("s")
        and imgWhite is not None
        and landmark_img is not None
        and counter < settings["max_samples"]
    ):
        try:
            timestamp = int(time.time() * 1000)

            # Save processed hand image
            img_filename = (
                f"{folder_images}/{gestures[current_gesture_index]}_{timestamp}.jpg"
            )
            cv2.imwrite(img_filename, imgWhite)

            # Save landmark skeleton image
            landmark_filename = (
                f"{folder_landmarks}/{gestures[current_gesture_index]}_{timestamp}.jpg"
            )
            cv2.imwrite(landmark_filename, landmark_img)

            # Save landmark coordinates as text file for ML training
            coords_filename = (
                f"{folder_landmarks}/{gestures[current_gesture_index]}_{timestamp}.txt"
            )
            with open(coords_filename, "w") as f:
                for lm in normalized_landmarks:
                    f.write(f"{lm[0]},{lm[1]}\n")

            counter += 1
            last_action_time = current_time
            confirmation_time = time.time()
            print(
                f"Saved {counter}/{settings['max_samples']} (image + landmarks) for {gestures[current_gesture_index]}"
            )

        except Exception as e:
            print(f"Capture failed: {str(e)}")

    elif key == ord("n"):
        skip_next = True
        print(f"Skipping {gestures[current_gesture_index]}")

    elif key == ord("q"):
        break

    cv2.imshow("Punjabi Sign Language Collector", img_with_hands)

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
print("Resources released. Enhanced Punjabi dataset collection completed!")
print(f"Data saved in:")
print(f"  - Images: Punjabi_Data/images/")
print(f"  - Landmarks: Punjabi_Data/landmarks/")
print(f"  - Coordinate files: Available as .txt files alongside landmark images")