import os
import cv2
import numpy as np
import tensorflow as tf
import json
import time
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
# Model configuration - adjust these paths based on your setup
MODEL_DIR = "Punjabi_Model_Optimized"
MODEL_FILE = "punjabi_landmark_model_optimized.keras"  # Your model file name
LABELS_FILE = "punjabi_landmark_model_optimized_labels.txt"  # Your labels file

# Data configuration
DATA_DIR = "Punjabi_Data_Augmented"
MAX_SAMPLES_PER_CLASS = 3  # Reduce for quick testing

# Test settings
VERBOSE = True

# ==============================
# UTILITY FUNCTIONS
# ==============================


def find_model_file():
    """Find the model file in the directory"""
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory '{MODEL_DIR}' not found!")
        return None, None, None

    # Look for .keras files
    keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

    if not keras_files:
        print(f"❌ No .keras model files found in '{MODEL_DIR}'!")
        print("Available files:")
        for f in os.listdir(MODEL_DIR):
            print(f"  - {f}")
        return None, None, None

    # Use the first .keras file found
    model_file = keras_files[0]
    model_path = os.path.join(MODEL_DIR, model_file)

    # Look for corresponding labels file
    base_name = model_file.replace(".keras", "")
    labels_file = f"{base_name}_labels.txt"
    labels_path = os.path.join(MODEL_DIR, labels_file)

    # If specific labels file not found, look for any .txt file
    if not os.path.exists(labels_path):
        txt_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".txt")]
        if txt_files:
            labels_path = os.path.join(MODEL_DIR, txt_files[0])
            print(f"⚠  Using labels file: {txt_files[0]}")
        else:
            print(f"❌ No labels file found!")
            return model_path, None, model_file

    return model_path, labels_path, model_file


def load_labels(labels_path):
    """Load gesture labels from file"""
    if not labels_path or not os.path.exists(labels_path):
        return None

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    # Handle different label formats
                    parts = line.split(maxsplit=1)
                    if len(parts) >= 2:
                        labels.append(parts[1])  # Use label after index
                    else:
                        labels.append(parts[0])  # Use the whole line
        return labels
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        return None


def get_gesture_classes():
    """Get available gesture classes from data directory"""
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found!")
        return []

    # Check for enhanced structure with landmarks folder
    landmarks_dir = os.path.join(DATA_DIR, "landmarks")
    if os.path.exists(landmarks_dir):
        gesture_folders = [
            d
            for d in os.listdir(landmarks_dir)
            if os.path.isdir(os.path.join(landmarks_dir, d))
        ]
        print(f"✅ Found landmarks data structure with {len(gesture_folders)} classes")
        return sorted(gesture_folders), "landmarks"

    # Check for images folder (but we'll look for coordinate files)
    images_dir = os.path.join(DATA_DIR, "images")
    if os.path.exists(images_dir):
        gesture_folders = [
            d
            for d in os.listdir(images_dir)
            if os.path.isdir(os.path.join(images_dir, d))
        ]
        print(f"✅ Found images data structure with {len(gesture_folders)} classes")
        return sorted(gesture_folders), "images"

    # Check for legacy structure (direct class folders)
    gesture_folders = [
        d
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in ["landmarks", "images"]
    ]
    print(f"✅ Found legacy data structure with {len(gesture_folders)} classes")
    return sorted(gesture_folders), "legacy"


def load_coordinate_data(coord_path):
    """Load landmark coordinates from text file"""
    try:
        coords = []
        with open(coord_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and "," in line:
                    try:
                        x, y = map(float, line.split(","))
                        coords.extend([x, y])
                    except ValueError:
                        continue

        if len(coords) == 42:  # 21 landmarks * 2 coordinates
            coords_array = np.array(coords, dtype=np.float32)

            # Apply normalization (similar to training)
            # Center the coordinates
            coords_centered = coords_array - np.median(coords_array)

            # Scale using interquartile range for robustness
            q75, q25 = np.percentile(coords_array, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                coords_scaled = coords_centered / iqr
            else:
                coords_scaled = coords_centered

            return coords_scaled.reshape(1, -1)
        else:
            if VERBOSE:
                print(f"    ⚠  Invalid coordinate count: {len(coords)} (expected 42)")
            return None
    except Exception as e:
        if VERBOSE:
            print(f"    ❌ Error loading coordinates from {coord_path}: {e}")
        return None


def get_coordinate_files(class_name, data_structure, max_samples=MAX_SAMPLES_PER_CLASS):
    """Get coordinate files for testing"""
    coordinate_files = []

    # Determine where to look for coordinate files
    if data_structure == "landmarks":
        class_dir = os.path.join(DATA_DIR, "landmarks", class_name)
    elif data_structure == "images":
        # Sometimes coordinate files might be in a separate landmarks folder
        landmarks_dir = os.path.join(DATA_DIR, "landmarks", class_name)
        if os.path.exists(landmarks_dir):
            class_dir = landmarks_dir
        else:
            # Or they might be alongside images
            class_dir = os.path.join(DATA_DIR, "images", class_name)
    else:  # legacy
        class_dir = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(class_dir):
        if VERBOSE:
            print(f"    ⚠  Directory not found: {class_dir}")
        return coordinate_files

    # Get .txt files (coordinate files)
    txt_files = [f for f in os.listdir(class_dir) if f.endswith(".txt")]

    # Limit samples
    txt_files = txt_files[:max_samples]

    for txt_file in txt_files:
        coordinate_files.append(os.path.join(class_dir, txt_file))

    return coordinate_files


def test_model_on_class(model, class_name, labels, data_structure):
    """Test model on coordinate samples from a specific class"""
    coord_files = get_coordinate_files(class_name, data_structure)

    if not coord_files:
        if VERBOSE:
            print(f"    ⚠  No coordinate files found for {class_name}")
        return {"tested": 0, "correct": 0, "samples": []}

    results = {"tested": 0, "correct": 0, "samples": []}

    for coord_path in coord_files:
        # Load coordinate data
        coord_input = load_coordinate_data(coord_path)
        if coord_input is None:
            continue

        try:
            # Make prediction
            predictions = model.predict(coord_input, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]

            # Get predicted label
            if labels and predicted_idx < len(labels):
                predicted_label = labels[predicted_idx]
            else:
                predicted_label = f"Class_{predicted_idx}"

            # Check if correct
            is_correct = predicted_label == class_name

            results["tested"] += 1
            if is_correct:
                results["correct"] += 1

            sample_result = {
                "file": os.path.basename(coord_path),
                "predicted": predicted_label,
                "confidence": float(confidence),
                "correct": is_correct,
            }
            results["samples"].append(sample_result)

            if VERBOSE:
                status = "✅" if is_correct else "❌"
                print(
                    f"    {status} {os.path.basename(coord_path)} → {predicted_label} ({confidence:.3f})"
                )

        except Exception as e:
            if VERBOSE:
                print(f"    ❌ Error with {os.path.basename(coord_path)}: {e}")
            continue

    return results


def check_model_input_type(model):
    """Check what type of input the model expects"""
    input_shape = model.input_shape
    if len(input_shape) == 2 and input_shape[1] == 42:
        return "coordinates"  # Landmark coordinates (21 points × 2)
    elif len(input_shape) == 4 and input_shape[1:3] == (300, 300):
        return "images"  # Images (300×300 RGB)
    else:
        return "unknown"


def run_smoke_test():
    """Run the complete smoke test"""
    print("🔥 SIMPLE PUNJABI SIGN LANGUAGE SMOKE TEST")
    print("=" * 60)

    # Find model
    print("\n📁 Finding model files...")
    model_path, labels_path, model_name = find_model_file()

    if not model_path:
        return

    print(f"✅ Found model: {model_name}")
    if labels_path:
        print(f"✅ Found labels: {os.path.basename(labels_path)}")

    # Load model
    print("\n🤖 Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")

        # Check input type
        input_type = check_model_input_type(model)
        print(f"   Input type: {input_type}")

        if input_type == "coordinates":
            print("   🎯 This is a LANDMARK-BASED model (expects coordinate data)")
        elif input_type == "images":
            print("   🖼  This is an IMAGE-BASED model (expects image data)")
        else:
            print("   ❓ Unknown input type - will try coordinate data")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load labels
    labels = load_labels(labels_path)
    if labels:
        print(f"✅ Loaded {len(labels)} gesture labels")
        if len(labels) <= 10:
            print(f"   Labels: {', '.join(labels)}")
        else:
            print(f"   Labels: {', '.join(labels[:5])}... (+{len(labels)-5} more)")
    else:
        print("⚠  No labels loaded - will use class indices")

    # Get gesture classes
    print("\n📂 Finding gesture classes...")
    result = get_gesture_classes()
    if not result:
        return

    gesture_classes, data_structure = result
    print(f"   Data structure: {data_structure}")
    print(
        f"   Classes: {', '.join(gesture_classes[:5])}"
        + (f"... (+{len(gesture_classes)-5} more)" if len(gesture_classes) > 5 else "")
    )

    # Run tests
    print(f"\n🧪 Testing model on {len(gesture_classes)} classes...")
    print(f"   Max samples per class: {MAX_SAMPLES_PER_CLASS}")
    print(f"   Looking for coordinate (.txt) files...")
    print("-" * 40)

    total_tested = 0
    total_correct = 0
    class_results = {}

    start_time = time.time()

    for i, class_name in enumerate(gesture_classes, 1):
        if VERBOSE:
            print(f"\n[{i:2d}/{len(gesture_classes)}] Testing: {class_name}")

        result = test_model_on_class(model, class_name, labels, data_structure)

        if result["tested"] > 0:
            class_accuracy = result["correct"] / result["tested"]
            class_results[class_name] = {
                "accuracy": class_accuracy,
                "tested": result["tested"],
                "correct": result["correct"],
            }

            total_tested += result["tested"]
            total_correct += result["correct"]

            if VERBOSE:
                print(
                    f"  📊 Accuracy: {class_accuracy*100:.1f}% ({result['correct']}/{result['tested']})"
                )
        else:
            if VERBOSE:
                print(f"  ⚠  No valid coordinate files found")

    test_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("🎯 SMOKE TEST RESULTS")
    print(f"{'='*60}")

    if total_tested > 0:
        overall_accuracy = total_correct / total_tested
        print(
            f"Overall Accuracy: {overall_accuracy*100:.2f}% ({total_correct}/{total_tested})"
        )
        print(f"Classes Tested: {len(class_results)}/{len(gesture_classes)}")
        print(f"Test Time: {test_time:.2f} seconds")
        print(f"Avg Time per Sample: {test_time/total_tested*1000:.1f}ms")

        # Show best and worst classes
        if class_results:
            sorted_classes = sorted(
                class_results.items(), key=lambda x: x[1]["accuracy"], reverse=True
            )

            print(f"\n🏆 Best Performing Classes:")
            for class_name, result in sorted_classes[:3]:
                print(
                    f"  • {class_name}: {result['accuracy']*100:.1f}% ({result['correct']}/{result['tested']})"
                )

            print(f"\n📉 Worst Performing Classes:")
            for class_name, result in sorted_classes[-3:]:
                print(
                    f"  • {class_name}: {result['accuracy']*100:.1f}% ({result['correct']}/{result['tested']})"
                )

        if overall_accuracy > 0.8:
            print(f"\n🎉 Great! Landmark model is performing well!")
        elif overall_accuracy > 0.6:
            print(f"\n👍 Good! Landmark model shows reasonable performance!")
        else:
            print(f"\n⚠  Landmark model may need more training or tuning.")

    else:
        print("❌ No coordinate samples were successfully tested!")
        print("\n🔍 Debugging info:")
        print(f"   • Model expects input shape: {model.input_shape}")
        print(f"   • Data directory: {DATA_DIR}")
        print(f"   • Data structure detected: {data_structure}")

        # Show what files are available
        if data_structure == "landmarks":
            landmarks_dir = os.path.join(DATA_DIR, "landmarks")
            if os.path.exists(landmarks_dir):
                sample_class = gesture_classes[0] if gesture_classes else None
                if sample_class:
                    sample_dir = os.path.join(landmarks_dir, sample_class)
                    if os.path.exists(sample_dir):
                        files = os.listdir(sample_dir)[:5]
                        print(f"   • Sample files in {sample_class}: {files}")

    print(f"\n✅ Smoke test completed!")


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    run_smoke_test()