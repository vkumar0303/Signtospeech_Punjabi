import cv2
import numpy as np
import time
import threading
from queue import Queue
import subprocess
import tensorflow as tf
import json
import os
from collections import deque, Counter
import mediapipe as mp

# ========== OPTIMIZED CONFIGURATION ==========
# Model Configuration
MODEL_DIR = "Punjabi_Model_Optimized"
MODEL_NAME = "punjabi_landmark_model_optimized"
IMG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5

# Prediction Configuration
PREDICTION_STABILITY_FRAMES = 8
GESTURE_HOLD_TIME = 1.2
PAUSE_THRESHOLD = 2.5
PREDICTION_COOLDOWN = 0.8

# Performance Optimization
USE_GPU_ACCELERATION = True
FRAME_SKIP = 2
RESIZE_FACTOR = 0.8

# Language Processing
MAX_PREDICTIONS = 5
WORD_COMPLETION_ENABLED = False  # Disabled due to speller issues
# ===============================================

# ========== GPU OPTIMIZATION ==========
if USE_GPU_ACCELERATION:
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU acceleration enabled")
        else:
            print("⚠ No GPU found, using CPU")
    except Exception as e:
        print(f"⚠ GPU configuration failed: {e}")

# ========== AUDIO SYSTEM ==========
print("🔊 Initializing audio system...")
try:
    subprocess.run(
        ["osascript", "-e", "set volume output volume 75"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["say", "-v", "Samantha", "Punjabi sign language system ready"], check=True
    )
    print("✓ Audio system initialized")
except Exception as e:
    print(f"⚠ Audio system warning: {e}")


class OptimizedSpeechEngine:
    """Enhanced speech engine with fallback options"""

    def _init_(self):
        self.queue = Queue()
        self.active = True
        self.speaking = False
        self.start_worker()

    def start_worker(self):
        def worker():
            while self.active:
                try:
                    text = self.queue.get(timeout=1)
                    if text is None:
                        break

                    self.speaking = True

                    try:
                        subprocess.run(
                            ["say", "-v", "Samantha", "-r", "160", text],
                            check=True,
                            capture_output=True,
                            timeout=10,
                        )
                        print(f"🔊 Spoken: {text}")
                    except Exception as e:
                        print(f"❌ Speech failed: {text} - {e}")

                    self.speaking = False

                except Exception as e:
                    self.speaking = False
                    if str(e) != "":
                        print(f"Speech error: {e}")

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def speak(self, text):
        if text and text.strip() and not self.speaking:
            self.queue.put(str(text).strip())

    def is_speaking(self):
        return self.speaking

    def stop(self):
        self.active = False
        self.queue.put(None)


tts = OptimizedSpeechEngine()


# ========== OPTIMIZED MODEL LOADER ==========
class OptimizedPunjabiClassifier:
    """Optimized classifier for Punjabi sign language"""

    def _init_(self, model_dir, model_name):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = None
        self.labels = []
        self.config = {}
        self.training_mode = None
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

        self.load_model_and_config()
        self.setup_hand_detection()

    def load_model_and_config(self):
        """Load the optimized Punjabi model and configuration"""
        try:
            # Load configuration
            config_path = os.path.join(self.model_dir, f"{self.model_name}_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                print(f"✓ Loaded configuration: {config_path}")

            # Load model
            model_paths = [
                os.path.join(self.model_dir, f"{self.model_name}_best.keras"),
                os.path.join(self.model_dir, f"{self.model_name}_final.keras"),
                os.path.join(self.model_dir, f"{self.model_name}.h5"),
            ]

            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        self.model = tf.keras.models.load_model(
                            model_path, compile=False
                        )
                        print(f"✓ Loaded model: {model_path}")
                        break
                    except Exception as e:
                        print(f"⚠ Failed to load {model_path}: {e}")
                        continue

            if self.model is None:
                raise Exception("No valid model found!")

            # Load labels
            labels_path = os.path.join(self.model_dir, f"{self.model_name}_labels.txt")
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    self.labels = [
                        line.strip().split(" ", 1)[1] for line in f.readlines()
                    ]
            else:
                if "gestures" in self.config:
                    self.labels = self.config["gestures"]
                else:
                    raise Exception("No labels found!")

            self.training_mode = self.config.get("training_mode", "landmarks")
            print(f"✓ Training mode: {self.training_mode}")
            print(f"✓ Loaded {len(self.labels)} gesture classes")

        except Exception as e:
            raise Exception(f"Model loading failed: {e}")

    def setup_hand_detection(self):
        """Setup MediaPipe hand detection for landmark extraction"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_landmarks(self, img):
        """Extract hand landmarks from image"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y])
            return np.array(coords), landmarks

        return None, None

    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch

    def preprocess_landmarks(self, coords):
        """Preprocess landmark coordinates for model input"""
        if coords is None or len(coords) != 42:
            return None

        coords_centered = coords - np.median(coords)
        coords_scaled = coords_centered / (
            np.percentile(coords, 75) - np.percentile(coords, 25) + 1e-8
        )
        coords_batch = np.expand_dims(coords_scaled, axis=0)
        return coords_batch

    def get_prediction(self, img, hand_landmarks=None):
        """Get prediction from the model with confidence scoring"""
        try:
            if self.training_mode == "landmarks":
                if hand_landmarks is None:
                    coords, hand_landmarks = self.extract_landmarks(img)
                else:
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y])
                    coords = np.array(coords)

                if coords is None:
                    return None, 0.0, None

                processed_coords = self.preprocess_landmarks(coords)
                if processed_coords is None:
                    return None, 0.0, None

                predictions = self.model.predict(processed_coords, verbose=0)
            else:
                processed_img = self.preprocess_image(img)
                predictions = self.model.predict(processed_img, verbose=0)

            confidence = np.max(predictions[0])
            predicted_class = np.argmax(predictions[0])
            gesture_name = self.labels[predicted_class]

            # Apply smoothing
            self.prediction_buffer.append((gesture_name, confidence))

            if len(self.prediction_buffer) >= 3:
                recent_predictions = list(self.prediction_buffer)[-3:]
                gesture_counts = Counter([pred[0] for pred in recent_predictions])
                most_common = gesture_counts.most_common(1)[0]

                if most_common[1] >= 2:
                    smoothed_gesture = most_common[0]
                    avg_confidence = np.mean(
                        [
                            pred[1]
                            for pred in recent_predictions
                            if pred[0] == smoothed_gesture
                        ]
                    )
                    return smoothed_gesture, avg_confidence, predictions[0]

            return gesture_name, confidence, predictions[0]

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None


# Initialize classifier
try:
    classifier = OptimizedPunjabiClassifier(MODEL_DIR, MODEL_NAME)
    print("✓ Punjabi sign language classifier ready")
except Exception as e:
    print(f"❌ Classifier initialization failed: {e}")
    exit(1)


# ========== OPTIMIZED REAL-TIME PROCESSOR ==========
class PunjabiSignProcessor:
    """Main processor for real-time Punjabi sign language recognition"""

    def _init_(self):
        self.current_word = []
        self.sentence = []
        self.predictions = []

        # Gesture tracking
        self.last_gesture = None
        self.gesture_start_time = 0
        self.gesture_confirmed = False
        self.last_prediction_time = 0
        self.stable_predictions = deque(maxlen=PREDICTION_STABILITY_FRAMES)
        self.no_hand_start_time = 0

        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def process_gesture(self, gesture, confidence):
        """Process detected gesture with stability checking"""
        current_time = time.time()

        self.stable_predictions.append(gesture)

        if len(self.stable_predictions) >= PREDICTION_STABILITY_FRAMES:
            recent_gestures = list(self.stable_predictions)[
                -PREDICTION_STABILITY_FRAMES:
            ]
            gesture_counts = Counter(recent_gestures)
            most_common = gesture_counts.most_common(1)[0]

            if most_common[1] >= PREDICTION_STABILITY_FRAMES * 0.7:
                stable_gesture = most_common[0]

                if stable_gesture != self.last_gesture:
                    self.last_gesture = stable_gesture
                    self.gesture_start_time = current_time
                    self.gesture_confirmed = False

                if (
                    current_time - self.gesture_start_time >= GESTURE_HOLD_TIME
                    and not self.gesture_confirmed
                    and current_time - self.last_prediction_time >= PREDICTION_COOLDOWN
                ):
                    self.execute_gesture(stable_gesture)
                    self.gesture_confirmed = True
                    self.last_prediction_time = current_time

    def execute_gesture(self, gesture):
        """Execute the confirmed gesture action"""
        print(f"Executing gesture: {gesture}")

        if gesture.lower() in ["space"]:
            self.add_space()
        elif gesture.lower() in ["delete"]:
            self.delete_character()
        elif gesture.lower() in ["comma"]:
            self.add_character(",")
            tts.speak("comma")
        elif gesture.lower() in ["period", "full_stop"]:
            self.add_character(".")
            tts.speak("period")
        elif gesture.lower() in ["question"]:
            self.add_character("?")
            tts.speak("question mark")
        else:
            self.add_character(gesture)
            tts.speak(gesture)

    def add_character(self, char):
        """Add character to current word"""
        self.current_word.append(char)

    def add_space(self):
        """Complete current word and add to sentence"""
        if self.current_word:
            completed_word = "".join(self.current_word)
            self.sentence.append(completed_word)
            tts.speak(completed_word)
            self.current_word = []

    def delete_character(self):
        """Delete last character"""
        if self.current_word:
            deleted = self.current_word.pop()
            tts.speak(f"deleted {deleted}")

    def handle_no_hands(self):
        """Handle periods when no hands are detected"""
        current_time = time.time()

        if self.no_hand_start_time == 0:
            self.no_hand_start_time = current_time
        elif current_time - self.no_hand_start_time >= PAUSE_THRESHOLD:
            if self.current_word:
                self.add_space()
            self.no_hand_start_time = 0

    def reset_no_hand_timer(self):
        """Reset the no-hand timer when hands are detected"""
        self.no_hand_start_time = 0

    def clear_all(self):
        """Clear current word and sentence"""
        self.current_word = []
        self.sentence = []
        self.predictions = []
        tts.speak("cleared")

    def speak_sentence(self):
        """Speak the complete sentence"""
        if self.sentence:
            full_text = " ".join(self.sentence)
            tts.speak(full_text)
        elif self.current_word:
            partial_text = "".join(self.current_word)
            tts.speak(partial_text)

    def get_display_text(self):
        """Get text for display"""
        current_text = "".join(self.current_word)
        sentence_text = " ".join(self.sentence)
        return current_text, sentence_text

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

        return self.fps_counter


processor = PunjabiSignProcessor()


# ========== UI DRAWING FUNCTIONS ==========
def draw_enhanced_interface(img, processor, gesture_info=None):
    """Draw enhanced user interface"""
    h, w = img.shape[:2]

    # Main panel
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (50, 50, 50), cv2.FILLED)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Title
    cv2.putText(
        img,
        "Punjabi Sign Language Translator",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    # Current input
    current_text, sentence_text = processor.get_display_text()
    cv2.putText(
        img,
        f"Current: {current_text}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    # Sentence
    if sentence_text:
        max_chars = 60
        if len(sentence_text) > max_chars:
            sentence_display = sentence_text[-max_chars:] + "..."
        else:
            sentence_display = sentence_text

        cv2.putText(
            img,
            f"Sentence: {sentence_display}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Gesture information
    if gesture_info:
        gesture, confidence = gesture_info
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(
            img,
            f"Detected: {gesture} ({confidence:.2f})",
            (w - 350, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # Control panel
    control_panel_y = h - 80

    # Clear button
    cv2.rectangle(img, (20, control_panel_y), (120, h - 20), (0, 0, 200), cv2.FILLED)
    cv2.putText(
        img,
        "Clear (C)",
        (30, control_panel_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Speak button
    cv2.rectangle(img, (140, control_panel_y), (240, h - 20), (0, 200, 0), cv2.FILLED)
    cv2.putText(
        img,
        "Speak (V)",
        (150, control_panel_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Space button
    cv2.rectangle(img, (260, control_panel_y), (360, h - 20), (200, 100, 0), cv2.FILLED)
    cv2.putText(
        img,
        "Space (S)",
        (270, control_panel_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Performance info
    fps = processor.update_fps()
    cv2.putText(
        img,
        f"FPS: {fps}",
        (w - 100, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def draw_hand_landmarks(img, landmarks):
    """Draw hand landmarks on image"""
    if landmarks:
        classifier.mp_draw.draw_landmarks(
            img,
            landmarks,
            classifier.mp_hands.HAND_CONNECTIONS,
            classifier.mp_draw.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            classifier.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
        )


# ========== MAIN APPLICATION LOOP ==========
def main():
    """Main application loop with optimized processing"""
    print("🚀 Starting Punjabi Sign Language Real-Time Converter")
    print("=" * 60)
    print("Controls:")
    print("  C - Clear all text")
    print("  V - Speak current sentence")
    print("  S - Add space manually")
    print("  Q - Quit application")
    print("=" * 60)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("✓ Camera initialized")
    tts.speak("Punjabi sign language system ready")

    frame_skip_counter = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("⚠ Failed to read frame")
                continue

            # Flip and resize for performance
            frame = cv2.flip(frame, 1)
            if RESIZE_FACTOR < 1.0:
                new_width = int(frame.shape[1] * RESIZE_FACTOR)
                new_height = int(frame.shape[0] * RESIZE_FACTOR)
                frame = cv2.resize(frame, (new_width, new_height))

            frame_skip_counter += 1

            # Process every nth frame for performance
            if frame_skip_counter % FRAME_SKIP == 0:
                gesture, confidence, raw_predictions = classifier.get_prediction(frame)

                if gesture and confidence > CONFIDENCE_THRESHOLD:
                    processor.process_gesture(gesture, confidence)
                    processor.reset_no_hand_timer()
                    gesture_info = (gesture, confidence)

                    # Draw landmarks if available
                    _, landmarks = classifier.extract_landmarks(frame)
                    draw_hand_landmarks(frame, landmarks)
                else:
                    processor.handle_no_hands()
                    gesture_info = None
            else:
                gesture_info = None

            # Draw interface
            draw_enhanced_interface(frame, processor, gesture_info)

            # Display
            cv2.imshow("Punjabi Sign Language Translator", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                processor.clear_all()
            elif key == ord("v"):
                processor.speak_sentence()
            elif key == ord("s"):
                processor.add_space()
            elif key == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")

    except Exception as e:
        print(f"❌ Error in main loop: {e}")

    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("✅ Application shutdown complete")


if __name__ == "__main__":
    main()