import cv2
import numpy as np
import os
import random
import math
import time
from PIL import Image, ImageEnhance, ImageFilter
import json

# ==============================
# CONFIGURATION
# ==============================
imgSize = 300

# Augmentation settings
augmentation_settings = {
    "rotation_range": (-15, 15),  # Degrees
    "scale_range": (0.8, 1.2),  # Scale factor
    "translation_range": (-30, 30),  # Pixels
    "brightness_range": (0.7, 1.3),  # Brightness factor
    "contrast_range": (0.8, 1.2),  # Contrast factor
    "noise_intensity": (0, 25),  # Gaussian noise
    "blur_range": (0, 2),  # Gaussian blur radius
    "flip_horizontal": True,  # Enable horizontal flip
    "augmentations_per_sample": 5,  # Number of augmented versions per original
}

# Punjabi Gurmukhi alphabet gestures (matching original)
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
# UTILITY FUNCTIONS
# ==============================
def load_landmarks_from_txt(filepath):
    """Load normalized landmarks from text file"""
    landmarks = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    x, y = map(float, line.strip().split(","))
                    landmarks.append([x, y])
        return landmarks
    except Exception as e:
        print(f"Error loading landmarks from {filepath}: {e}")
        return None


def save_landmarks_to_txt(landmarks, filepath):
    """Save normalized landmarks to text file"""
    try:
        with open(filepath, "w") as f:
            for lm in landmarks:
                f.write(f"{lm[0]},{lm[1]}\n")
        return True
    except Exception as e:
        print(f"Error saving landmarks to {filepath}: {e}")
        return False


def draw_landmarks_on_white_bg(landmarks, img_size=300):
    """Draw hand landmarks skeleton on white background (matching original)"""
    # Create white background
    landmark_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    if not landmarks:
        return landmark_img

    # Convert normalized landmarks to image coordinates
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


# ==============================
# AUGMENTATION FUNCTIONS
# ==============================
def augment_landmarks(landmarks, transformation_params):
    """Apply transformations to landmarks"""
    if not landmarks:
        return landmarks

    augmented_landmarks = []

    # Extract transformation parameters
    rotation_angle = transformation_params.get("rotation", 0)
    scale_factor = transformation_params.get("scale", 1.0)
    tx = transformation_params.get("tx", 0)
    ty = transformation_params.get("ty", 0)
    flip_h = transformation_params.get("flip_horizontal", False)

    # Center point for rotation and scaling
    center_x, center_y = 0.5, 0.5

    for lm in landmarks:
        x, y = lm[0], lm[1]

        # Apply horizontal flip
        if flip_h:
            x = 1.0 - x

        # Translate to center for rotation and scaling
        x_centered = x - center_x
        y_centered = y - center_y

        # Apply rotation
        if rotation_angle != 0:
            angle_rad = math.radians(rotation_angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            x_rotated = x_centered * cos_angle - y_centered * sin_angle
            y_rotated = x_centered * sin_angle + y_centered * cos_angle
        else:
            x_rotated = x_centered
            y_rotated = y_centered

        # Apply scaling
        x_scaled = x_rotated * scale_factor
        y_scaled = y_rotated * scale_factor

        # Translate back and apply translation
        x_final = x_scaled + center_x + (tx / imgSize)
        y_final = y_scaled + center_y + (ty / imgSize)

        # Clamp to valid range
        x_final = max(0.0, min(1.0, x_final))
        y_final = max(0.0, min(1.0, y_final))

        augmented_landmarks.append([x_final, y_final])

    return augmented_landmarks


def augment_image(image, transformation_params):
    """Apply transformations to image"""
    if image is None:
        return None

    # Convert to PIL for easier transformations
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Apply brightness
    brightness = transformation_params.get("brightness", 1.0)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)

    # Apply contrast
    contrast = transformation_params.get("contrast", 1.0)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)

    # Apply blur
    blur_radius = transformation_params.get("blur", 0)
    if blur_radius > 0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert back to numpy array
    img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Apply geometric transformations
    rows, cols = img_array.shape[:2]

    # Rotation
    rotation_angle = transformation_params.get("rotation", 0)
    if rotation_angle != 0:
        M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        img_array = cv2.warpAffine(
            img_array,
            M_rot,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    # Scale and Translation
    scale_factor = transformation_params.get("scale", 1.0)
    tx = transformation_params.get("tx", 0)
    ty = transformation_params.get("ty", 0)

    if scale_factor != 1.0 or tx != 0 or ty != 0:
        M_trans = np.float32(
            [
                [scale_factor, 0, tx + cols * (1 - scale_factor) / 2],
                [0, scale_factor, ty + rows * (1 - scale_factor) / 2],
            ]
        )
        img_array = cv2.warpAffine(
            img_array,
            M_trans,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    # Horizontal flip
    flip_h = transformation_params.get("flip_horizontal", False)
    if flip_h:
        img_array = cv2.flip(img_array, 1)

    # Add noise
    noise_intensity = transformation_params.get("noise", 0)
    if noise_intensity > 0:
        noise = np.random.normal(0, noise_intensity, img_array.shape).astype(np.uint8)
        img_array = cv2.add(img_array, noise)

    return img_array


def generate_transformation_params():
    """Generate random transformation parameters"""
    params = {}

    # Rotation
    params["rotation"] = random.uniform(*augmentation_settings["rotation_range"])

    # Scale
    params["scale"] = random.uniform(*augmentation_settings["scale_range"])

    # Translation
    params["tx"] = random.randint(*augmentation_settings["translation_range"])
    params["ty"] = random.randint(*augmentation_settings["translation_range"])

    # Brightness
    params["brightness"] = random.uniform(*augmentation_settings["brightness_range"])

    # Contrast
    params["contrast"] = random.uniform(*augmentation_settings["contrast_range"])

    # Noise
    params["noise"] = random.randint(*augmentation_settings["noise_intensity"])

    # Blur
    params["blur"] = random.uniform(*augmentation_settings["blur_range"])

    # Horizontal flip (50% chance)
    params["flip_horizontal"] = (
        random.choice([True, False])
        if augmentation_settings["flip_horizontal"]
        else False
    )

    return params


# ==============================
# MAIN AUGMENTATION FUNCTION
# ==============================
def augment_gesture_data(gesture_name):
    """Augment data for a specific gesture"""
    print(f"\nProcessing gesture: {gesture_name}")

    # Define paths
    original_images_path = f"Punjabi_Data/images/{gesture_name}"
    original_landmarks_path = f"Punjabi_Data/landmarks/{gesture_name}"

    augmented_images_path = f"Punjabi_Data_Augmented/images/{gesture_name}"
    augmented_landmarks_path = f"Punjabi_Data_Augmented/landmarks/{gesture_name}"

    # Create augmented directories
    os.makedirs(augmented_images_path, exist_ok=True)
    os.makedirs(augmented_landmarks_path, exist_ok=True)

    # Check if original data exists
    if not os.path.exists(original_images_path) or not os.path.exists(
        original_landmarks_path
    ):
        print(f"Warning: Original data not found for {gesture_name}")
        return 0

    # Get list of original files
    image_files = [f for f in os.listdir(original_images_path) if f.endswith(".jpg")]

    augmented_count = 0

    for image_file in image_files:
        # Extract timestamp from filename
        timestamp = image_file.replace(f"{gesture_name}_", "").replace(".jpg", "")

        # Load original image
        original_image_path = os.path.join(original_images_path, image_file)
        original_image = cv2.imread(original_image_path)

        if original_image is None:
            print(f"Warning: Could not load {original_image_path}")
            continue

        # Load original landmarks
        landmarks_txt_path = os.path.join(
            original_landmarks_path, f"{gesture_name}_{timestamp}.txt"
        )
        original_landmarks = load_landmarks_from_txt(landmarks_txt_path)

        if original_landmarks is None:
            print(f"Warning: Could not load landmarks for {timestamp}")
            continue

        # Generate augmented versions
        for aug_idx in range(augmentation_settings["augmentations_per_sample"]):
            try:
                # Generate transformation parameters
                transform_params = generate_transformation_params()

                # Apply transformations
                augmented_image = augment_image(original_image, transform_params)
                augmented_landmarks = augment_landmarks(
                    original_landmarks, transform_params
                )

                if augmented_image is None or not augmented_landmarks:
                    continue

                # Generate landmark skeleton image
                landmark_skeleton = draw_landmarks_on_white_bg(
                    augmented_landmarks, imgSize
                )

                # Save augmented data
                aug_timestamp = f"{timestamp}aug{aug_idx}"

                # Save augmented hand image
                aug_image_path = os.path.join(
                    augmented_images_path, f"{gesture_name}_{aug_timestamp}.jpg"
                )
                cv2.imwrite(aug_image_path, augmented_image)

                # Save augmented landmark skeleton image
                aug_landmark_img_path = os.path.join(
                    augmented_landmarks_path, f"{gesture_name}_{aug_timestamp}.jpg"
                )
                cv2.imwrite(aug_landmark_img_path, landmark_skeleton)

                # Save augmented landmark coordinates
                aug_coords_path = os.path.join(
                    augmented_landmarks_path, f"{gesture_name}_{aug_timestamp}.txt"
                )
                save_landmarks_to_txt(augmented_landmarks, aug_coords_path)

                augmented_count += 1

            except Exception as e:
                print(f"Error processing {gesture_name}{timestamp}_aug{aug_idx}: {e}")
                continue

    print(f"Generated {augmented_count} augmented samples for {gesture_name}")
    return augmented_count


# ==============================
# MAIN EXECUTION
# ==============================
def main():
    print("=== Punjabi Sign Language Data Augmentation ===")
    print(
        f"Generating {augmentation_settings['augmentations_per_sample']} augmented versions per original sample"
    )
    print("Output: Images + Landmark skeletons + Coordinate files")

    # Create main augmented directory
    if not os.path.exists("Punjabi_Data_Augmented"):
        os.makedirs("Punjabi_Data_Augmented")
        print("Created Punjabi_Data_Augmented directory")

    # Create subdirectories
    for data_type in ["landmarks", "images"]:
        if not os.path.exists(f"Punjabi_Data_Augmented/{data_type}"):
            os.makedirs(f"Punjabi_Data_Augmented/{data_type}")

    total_augmented = 0
    start_time = time.time()

    # Process each gesture
    for i, gesture in enumerate(gestures):
        print(f"\nProgress: {i+1}/{len(gestures)}")
        augmented_count = augment_gesture_data(gesture)
        total_augmented += augmented_count

        # Show progress
        elapsed = time.time() - start_time
        if i > 0:
            eta = (elapsed / (i + 1)) * (len(gestures) - i - 1)
            print(f"ETA: {eta/60:.1f} minutes")

    # Summary
    print(f"\n=== AUGMENTATION COMPLETE ===")
    print(f"Total augmented samples generated: {total_augmented}")
    print(f"Total time taken: {(time.time() - start_time)/60:.2f} minutes")
    print(f"\nAugmented data saved in:")
    print(f"  - Images: Punjabi_Data_Augmented/images/")
    print(f"  - Landmark skeletons: Punjabi_Data_Augmented/landmarks/")
    print(f"  - Coordinate files: Available as .txt files alongside landmark images")

    # Statistics
    print(f"\nAugmentation settings used:")
    for key, value in augmentation_settings.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()

# Augment DataSet Meta:
# Total augmented samples generated: 54000
# Total time taken: 15.38 minutes

# Augmented data saved in:
#   - Images: Punjabi_Data_Augmented/images/
#   - Landmark skeletons: Punjabi_Data_Augmented/landmarks/
#   - Coordinate files: Available as .txt files alongside landmark images

# Augmentation settings used:
#   - rotation_range: (-15, 15)
#   - scale_range: (0.8, 1.2)
#   - translation_range: (-30, 30)
#   - brightness_range: (0.7, 1.3)
#   - contrast_range: (0.8, 1.2)
#   - noise_intensity: (0, 25)
#   - blur_range: (0, 2)
#   - flip_horizontal: True
#   - augmentations_per_sample: 5