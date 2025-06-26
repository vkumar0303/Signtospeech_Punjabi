import os
import shutil
import numpy as np
import cv2
import math
import time
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import seaborn as sns

# ==============================
# OPTIMIZED CONFIGURATION
# ==============================
# Data directory selection
AUGMENTED_DATA_DIR = "Punjabi_Data_Augmented"
ORIGINAL_DATA_DIR = "Punjabi_Data"

# Choose data source
USE_AUGMENTED_DATA = True

# Optimized model parameters
IMG_SIZE = 300
BATCH_SIZE = 16  # Smaller batch for better generalization
EPOCHS = 80  # Reduced from 100
LEARNING_RATE = 0.001  # Slightly higher initial learning rate

# Training mode selection
TRAINING_MODE = "landmarks"  # Focus on landmarks for efficiency
USE_COORDINATE_DATA = True

# Anti-overfitting settings
USE_CLASS_WEIGHTS = True
VALIDATION_SPLIT = 0.2  # Increased validation split
EARLY_STOPPING_PATIENCE = 12  # Reduced patience
REDUCE_LR_PATIENCE = 5  # More aggressive LR reduction
MIN_DELTA = 0.001  # Minimum improvement threshold

# Data augmentation intensity (reduced to prevent overfitting)
AUGMENTATION_INTENSITY = 0.7  # Scale factor for augmentation


# ==============================
# UTILITY FUNCTIONS (Optimized)
# ==============================
def select_data_directory():
    """Select appropriate data directory based on availability"""
    if USE_AUGMENTED_DATA and os.path.exists(AUGMENTED_DATA_DIR):
        print(f"✓ Using augmented dataset: {AUGMENTED_DATA_DIR}")
        return AUGMENTED_DATA_DIR
    elif os.path.exists(ORIGINAL_DATA_DIR):
        print(f"✓ Using original dataset: {ORIGINAL_DATA_DIR}")
        return ORIGINAL_DATA_DIR
    else:
        raise Exception("No valid data directory found! Run data collection first.")


def verify_data_structure(data_dir):
    """Verify the enhanced data structure exists"""
    if not os.path.exists(data_dir):
        raise Exception(f"Data directory '{data_dir}' not found.")

    has_images = os.path.exists(os.path.join(data_dir, "images"))
    has_landmarks = os.path.exists(os.path.join(data_dir, "landmarks"))
    has_old_structure = any(
        os.path.isdir(os.path.join(data_dir, d)) and d not in ["images", "landmarks"]
        for d in os.listdir(data_dir)
    )

    if has_images and has_landmarks:
        print("✓ Enhanced data structure detected (images + landmarks)")
        return "enhanced"
    elif has_old_structure:
        print("✓ Legacy data structure detected")
        return "legacy"
    else:
        raise Exception("No valid data structure found!")


def get_gesture_classes(data_dir, structure_type):
    """Get available gesture classes based on data structure"""
    if structure_type == "enhanced":
        if TRAINING_MODE in ["images", "both"]:
            base_dir = os.path.join(data_dir, "images")
        else:
            base_dir = os.path.join(data_dir, "landmarks")
    else:
        base_dir = data_dir

    gestures = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    return gestures


def analyze_dataset(gestures, data_dir, structure_type):
    """Analyze dataset composition and balance"""
    print(f"\nDataset Analysis:")
    print("-" * 40)

    class_counts = {}
    total_samples = 0

    for gesture in gestures:
        gesture_samples = 0

        if structure_type == "enhanced":
            if TRAINING_MODE in ["images", "both"]:
                img_dir = os.path.join(data_dir, "images", gesture)
                if os.path.exists(img_dir):
                    img_count = len(
                        [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
                    )
                    gesture_samples += img_count

            if TRAINING_MODE in ["landmarks", "both"]:
                landmark_dir = os.path.join(data_dir, "landmarks", gesture)
                if os.path.exists(landmark_dir):
                    landmark_count = len(
                        [
                            f
                            for f in os.listdir(landmark_dir)
                            if f.lower().endswith(".jpg")
                        ]
                    )
                    gesture_samples += landmark_count
        else:
            class_dir = os.path.join(data_dir, gesture)
            if os.path.exists(class_dir):
                gesture_samples = len(
                    [f for f in os.listdir(class_dir) if f.lower().endswith(".jpg")]
                )

        class_counts[gesture] = gesture_samples
        total_samples += gesture_samples

    # Display statistics
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    avg_samples = total_samples / len(gestures)

    print(f"Total samples: {total_samples:,}")
    print(f"Average per class: {avg_samples:.1f}")
    print(f"Min samples: {min_samples}")
    print(f"Max samples: {max_samples}")
    print(f"Class balance ratio: {min_samples/max_samples:.3f}")

    return class_counts


def calculate_class_weights(class_counts):
    """Calculate balanced class weights with smoothing"""
    if not USE_CLASS_WEIGHTS:
        return None

    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)

    class_weights = {}
    # Use smoothed class weights to prevent extreme values
    for i, (gesture, count) in enumerate(sorted(class_counts.items())):
        # Smooth the weight calculation
        weight = (total_samples / (num_classes * count)) ** 0.75  # Power smoothing
        weight = min(weight, 3.0)  # Cap maximum weight
        class_weights[i] = weight

    print(f"\nClass weights calculated for {num_classes} classes (smoothed)")
    return class_weights


def load_coordinate_data(landmarks_dir, gestures):
    """Load raw coordinate data for landmark-based training with filtering"""
    X_coords = []
    y_coords = []

    print("Loading coordinate data...")

    for gesture_idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(landmarks_dir, gesture)
        if not os.path.exists(gesture_dir):
            continue

        coord_files = [f for f in os.listdir(gesture_dir) if f.endswith(".txt")]
        loaded_count = 0

        for coord_file in coord_files:
            coord_path = os.path.join(gesture_dir, coord_file)
            try:
                coords = []
                with open(coord_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and "," in line:
                            x, y = map(float, line.split(","))
                            coords.extend([x, y])

                if len(coords) == 42:  # 21 landmarks * 2 coordinates
                    # Basic quality filtering
                    coords_array = np.array(coords).reshape(-1, 2)
                    if np.all(coords_array >= 0) and np.all(coords_array <= 1):
                        X_coords.append(coords)
                        y_coords.append(gesture_idx)
                        loaded_count += 1

            except Exception as e:
                print(f"Error loading {coord_path}: {e}")

        if loaded_count > 0:
            print(f"  {gesture}: {loaded_count} coordinate samples")

    return np.array(X_coords), np.array(y_coords)


# ==============================
# OPTIMIZED MODEL ARCHITECTURES
# ==============================
def create_optimized_image_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Create regularized CNN model to prevent overfitting"""
    model = keras.Sequential(
        [
            # Input layer
            keras.layers.Input(shape=input_shape),
            # First conv block - smaller filters, more regularization
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.15),
            # Second conv block
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.2),
            # Third conv block
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            # Fourth conv block - reduced size
            keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),  # Better than flatten
            # Classification head - simplified
            keras.layers.Dense(512, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def create_optimized_landmark_model(num_classes, input_shape=42):
    """Create regularized MLP model for landmark coordinates"""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_shape,)),
            # Input processing with L2 regularization
            keras.layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            # Hidden layers with progressive regularization
            keras.layers.Dense(
                256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(
                512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            # Bottleneck layer
            keras.layers.Dense(
                256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(
                128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.Dropout(0.3),
            # Output layer
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def create_moderate_data_augmentation():
    """Create moderate data augmentation to prevent overfitting"""
    return keras.Sequential(
        [
            keras.layers.RandomRotation(
                0.05 * AUGMENTATION_INTENSITY
            ),  # Reduced rotation
            keras.layers.RandomZoom(0.08 * AUGMENTATION_INTENSITY),  # Reduced zoom
            keras.layers.RandomTranslation(
                0.08 * AUGMENTATION_INTENSITY, 0.08 * AUGMENTATION_INTENSITY
            ),  # Reduced translation
            keras.layers.RandomBrightness(
                0.08 * AUGMENTATION_INTENSITY
            ),  # Reduced brightness
            keras.layers.RandomContrast(
                0.08 * AUGMENTATION_INTENSITY
            ),  # Reduced contrast
        ]
    )


def plot_training_metrics_enhanced(history, model_name):
    """Enhanced training metrics visualization with overfitting detection"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Accuracy plot
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(train_acc) + 1)

    axes[0, 0].plot(epochs, train_acc, "b-", label="Train Accuracy", linewidth=2)
    axes[0, 0].plot(epochs, val_acc, "r-", label="Validation Accuracy", linewidth=2)
    axes[0, 0].set_title(f"{model_name} - Accuracy", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Add best validation marker
    best_val_epoch = np.argmax(val_acc)
    axes[0, 0].axvline(
        x=best_val_epoch + 1,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best Val (Epoch {best_val_epoch+1})",
    )

    # Loss plot
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    axes[0, 1].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    axes[0, 1].plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)
    axes[0, 1].set_title(f"{model_name} - Loss", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Overfitting detection plot
    gap = np.array(train_acc) - np.array(val_acc)
    axes[0, 2].plot(epochs, gap, "purple", linewidth=2)
    axes[0, 2].axhline(
        y=0.05, color="red", linestyle="--", alpha=0.7, label="Overfitting Threshold"
    )
    axes[0, 2].set_title("Overfitting Detection", fontsize=14, fontweight="bold")
    axes[0, 2].set_ylabel("Train Acc - Val Acc")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Learning rate plot
    if "lr" in history.history:
        axes[1, 0].plot(epochs, history.history["lr"], linewidth=2, color="red")
        axes[1, 0].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale("log")
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Learning Rate\nData Not Available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=12,
        )
        axes[1, 0].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")

    # Validation accuracy improvement
    val_acc_smooth = np.convolve(
        val_acc, np.ones(5) / 5, mode="valid"
    )  # Moving average
    axes[1, 1].plot(
        range(3, len(val_acc_smooth) + 3),
        val_acc_smooth,
        "green",
        linewidth=2,
        label="Val Acc (Smoothed)",
    )
    axes[1, 1].plot(epochs, val_acc, "lightgreen", alpha=0.5, label="Val Acc (Raw)")
    axes[1, 1].set_title("Validation Accuracy Trend", fontsize=14, fontweight="bold")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Training summary with overfitting analysis
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    best_val_acc = max(val_acc)
    final_gap = final_train_acc - final_val_acc
    avg_gap = np.mean(gap[-10:])  # Average gap in last 10 epochs

    # Simple overfitting assessment
    if avg_gap > 0.1:
        overfitting_status = "🔴 High overfitting risk"
    elif avg_gap > 0.05:
        overfitting_status = "🟡 Moderate overfitting"
    else:
        overfitting_status = "🟢 Good generalization"

    summary_text = f"""Training Summary:

Final Train Accuracy: {final_train_acc:.4f}
Final Val Accuracy: {final_val_acc:.4f}
Best Val Accuracy: {best_val_acc:.4f}

Final Accuracy Gap: {final_gap:.4f}
Avg Gap (last 10): {avg_gap:.4f}

Overfitting Status: {overfitting_status}

Total Epochs: {len(train_acc)}
Best Epoch: {best_val_epoch + 1}
"""

    axes[1, 2].text(
        0.1,
        0.5,
        summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment="center",
        fontfamily="monospace",
    )
    axes[1, 2].set_title("Training Analysis", fontsize=14, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


# ==============================
# MAIN EXECUTION WITH OPTIMIZATIONS
# ==============================
def main():
    print("Optimized Punjabi Sign Language Model Trainer")
    print("=" * 60)

    # Select and verify data directory
    data_dir = select_data_directory()
    structure_type = verify_data_structure(data_dir)
    gestures = get_gesture_classes(data_dir, structure_type)

    print(f"Found {len(gestures)} Punjabi gesture classes")

    # Analyze dataset
    class_counts = analyze_dataset(gestures, data_dir, structure_type)
    class_weights = calculate_class_weights(class_counts)

    # Create output directory
    model_output_dir = "Punjabi_Model_Optimized"
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\nTraining Mode: {TRAINING_MODE}")
    print("=" * 40)

    if TRAINING_MODE == "images":
        # Image-based training with optimization
        print("Setting up optimized image-based training...")

        if structure_type == "enhanced":
            data_path = os.path.join(data_dir, "images")
        else:
            data_path = data_dir

        # Create data generators
        train_dataset = keras.utils.image_dataset_from_directory(
            data_path,
            validation_split=VALIDATION_SPLIT,
            subset="training",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        val_dataset = keras.utils.image_dataset_from_directory(
            data_path,
            validation_split=VALIDATION_SPLIT,
            subset="validation",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )

        # Moderate data augmentation
        data_augmentation = create_moderate_data_augmentation()
        normalization_layer = keras.layers.Rescaling(1.0 / 255)

        # Apply preprocessing
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(normalization_layer(x)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        val_dataset = val_dataset.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Optimize performance
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Create and compile model
        model = create_optimized_image_model(len(gestures))

        # Use adaptive learning rate with cosine decay
        initial_learning_rate = LEARNING_RATE
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate, first_decay_steps=20, t_mul=2.0, m_mul=0.9, alpha=0.1
        )

        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=lr_schedule, weight_decay=0.0001
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model_name = "punjabi_image_model_optimized"
        training_data = (train_dataset, val_dataset)

    elif TRAINING_MODE == "landmarks" and USE_COORDINATE_DATA:
        # Optimized coordinate-based training
        print("Setting up optimized landmark coordinate training...")

        landmarks_dir = os.path.join(data_dir, "landmarks")
        X_coords, y_coords = load_coordinate_data(landmarks_dir, gestures)

        if len(X_coords) == 0:
            raise Exception("No coordinate data found! Check your landmarks directory.")

        print(f"Loaded {len(X_coords)} coordinate samples")

        # Enhanced data preprocessing
        # Normalize with robust scaling
        X_coords_centered = X_coords - np.median(X_coords, axis=0)
        X_coords_scaled = X_coords_centered / (
            np.percentile(X_coords, 75, axis=0)
            - np.percentile(X_coords, 25, axis=0)
            + 1e-8
        )

        # Add slight noise for regularization
        noise = np.random.normal(0, 0.01, X_coords_scaled.shape)
        X_coords_augmented = X_coords_scaled + noise

        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X_coords_augmented,
            y_coords,
            test_size=VALIDATION_SPLIT,
            random_state=123,
            stratify=y_coords,
        )

        # Convert to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(gestures))
        y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(gestures))

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Create and compile model
        model = create_optimized_landmark_model(len(gestures))

        # Use adaptive learning rate
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            LEARNING_RATE, first_decay_steps=15, t_mul=2.0, m_mul=0.8, alpha=0.05
        )

        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=lr_schedule, weight_decay=0.0005
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model_name = "punjabi_landmark_model_optimized"
        training_data = ((X_train, y_train_cat), (X_val, y_val_cat))

    # Enhanced callbacks for preventing overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            min_delta=MIN_DELTA,
        ),
        keras.callbacks.ModelCheckpoint(
            f"{model_output_dir}/{model_name}_best.keras",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
            min_delta=MIN_DELTA,
        ),
        keras.callbacks.CSVLogger(f"{model_output_dir}/{model_name}_training_log.csv"),
    ]

    # Model summary
    print(f"\n{model_name.replace('_', ' ').title()} Architecture:")
    print("=" * 50)
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    # Training
    print(f"\nStarting optimized training with {EPOCHS} epochs...")
    print("=" * 50)

    start_time = time.time()

    if TRAINING_MODE == "landmarks" and USE_COORDINATE_DATA:
        history = model.fit(
            training_data[0][0],
            training_data[0][1],
            epochs=EPOCHS,
            validation_data=(training_data[1][0], training_data[1][1]),
            callbacks=callbacks,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            verbose=1,
        )
    else:
        history = model.fit(
            training_data[0],
            epochs=EPOCHS,
            validation_data=training_data[1],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    training_time = time.time() - start_time

    # Save models and configuration
    model.save(f"{model_output_dir}/{model_name}_final.keras")

    # Enhanced configuration
    config = {
        "training_mode": TRAINING_MODE,
        "use_coordinate_data": USE_COORDINATE_DATA,
        "use_augmented_data": USE_AUGMENTED_DATA,
        "data_directory": data_dir,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "validation_split": VALIDATION_SPLIT,
        "num_classes": len(gestures),
        "total_parameters": int(total_params),
        "training_time_seconds": training_time,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "augmentation_intensity": AUGMENTATION_INTENSITY,
        "min_delta": MIN_DELTA,
        "gestures": gestures,
        "structure_type": structure_type,
        "class_counts": class_counts,
        "optimization_features": [
            "L2 regularization",
            "Batch normalization",
            "Adaptive learning rate",
            "Weight decay (AdamW)",
            "Moderate augmentation",
            "Robust preprocessing",
            "Class weight balancing",
        ],
    }

    with open(f"{model_output_dir}/{model_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save labels
    with open(f"{model_output_dir}/{model_name}_labels.txt", "w") as f:
        for i, gesture in enumerate(gestures):
            f.write(f"{i} {gesture}\n")

    # Enhanced visualization
    fig = plot_training_metrics_enhanced(history, model_name.replace("_", " ").title())
    fig.savefig(
        f"{model_output_dir}/{model_name}_training_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Final evaluation
    print("\n" + "=" * 60)
    print("OPTIMIZED TRAINING COMPLETE!")
    print("=" * 60)

    final_val_accuracy = max(history.history["val_accuracy"])
    final_train_accuracy = history.history["accuracy"][-1]
    final_gap = final_train_accuracy - history.history["val_accuracy"][-1]
    best_epoch = np.argmax(history.history["val_accuracy"]) + 1

    print(f"Model type: {TRAINING_MODE}")
    print(f"Data source: {'Augmented' if USE_AUGMENTED_DATA else 'Original'}")
    print(f"Total classes: {len(gestures)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {final_val_accuracy:.4f} (epoch {best_epoch})")
    print(f"Final training accuracy: {final_train_accuracy:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final accuracy gap: {final_gap:.4f}")

    # Overfitting assessment
    if final_gap > 0.1:
        print("⚠  Warning: High overfitting detected! Consider more regularization.")
    elif final_gap > 0.05:
        print("🟡 Moderate overfitting - model should still generalize well.")
    else:
        print("🟢 Good generalization - low overfitting risk!")

    print(f"\nOptimized model files saved:")
    print(f"  - {model_output_dir}/{model_name}_best.keras")
    print(f"  - {model_output_dir}/{model_name}_final.keras")
    print(f"  - {model_output_dir}/{model_name}_labels.txt")
    print(f"  - {model_output_dir}/{model_name}_config.json")
    print(f"  - {model_output_dir}/{model_name}_training_analysis.png")
    print(f"  - {model_output_dir}/{model_name}_training_log.csv")

    return model, history, config


if __name__ == "__main__":
    model, history, config = main()