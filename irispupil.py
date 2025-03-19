# irispupil.py

import cv2
import os
import numpy as np
import torch

# -----------------------------
# Global Variables and Settings
# -----------------------------
# Default model path for YOLO iris & pupil detection (use a relative path)
default_model_path = "weights/best.pt"  # Ensure your weights file is in the 'weights' folder

# Class indices for iris and pupil (update according to your dataset)
IRIS_CLASS = 0   # Example: Class ID for iris
PUPIL_CLASS = 1  # Example: Class ID for pupil

# Directory to store cropped eye images (for detection)
detected_eyes_dir = "detected_eyes"
os.makedirs(detected_eyes_dir, exist_ok=True)

# -----------------------------
# Function 1: Eye Detection
# -----------------------------
# Initialize the eye detector.
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face_and_eyes(image, save_detected=False, person_counter=1):
    """
    Detects eyes in the provided image using Haar Cascade.
    Draws bounding boxes and labels for left/right eyes.
    
    Args:
        image (np.array): Input image in BGR format.
        save_detected (bool): If True, saves cropped eye images.
        person_counter (int): Counter used in filenames when saving images.
        
    Returns:
        annotated_image (np.array): The image annotated with detected eyes.
        eyes (list): A list of eye bounding boxes.
    """
    annotated_image = image.copy()
    gray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for (ex, ey, ew, eh) in eyes:
        eye_center_x = ex + ew // 2
        frame_center_x = annotated_image.shape[1] // 2

        if eye_center_x < frame_center_x:
            eye_label = "Left Eye"
            color = (0, 255, 0)  # Green for left eye.
        else:
            eye_label = "Right Eye"
            color = (255, 0, 0)  # Blue for right eye.

        cv2.rectangle(annotated_image, (ex, ey), (ex + ew, ey + eh), color, 2)
        cv2.putText(annotated_image, eye_label, (ex, ey - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_detected:
            eye_side = "left" if eye_center_x < frame_center_x else "right"
            eye_region = image[ey:ey+eh, ex:ex+ew]
            filename = f"person_{person_counter}_{eye_side}_eye.jpg"
            cv2.imwrite(os.path.join(detected_eyes_dir, filename), eye_region)

    return annotated_image, eyes

# -----------------------------
# Function 2: Image Enhancement
# -----------------------------
def remove_specular_reflection(image):
    """
    Removes specular reflections from a grayscale image using inpainting.
    """
    _, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted_image

def enhance_image(image):
    """
    Enhances the provided image using:
      - Specular reflection removal
      - CLAHE (Contrast Limited Adaptive Histogram Equalization)
      - Bilateral Filtering for noise reduction
      - Sharpening
    
    Args:
        image (np.array): Input image in BGR format.
    
    Returns:
        enhanced (np.array): Enhanced image (grayscale).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    no_reflection = remove_specular_reflection(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(no_reflection)
    bilateral_filtered = cv2.bilateralFilter(clahe_image, d=15, sigmaColor=100, sigmaSpace=100)
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(bilateral_filtered, -1, sharpening_kernel)
    return sharpened

# -----------------------------
# Function 3: Iris & Pupil Detection via YOLO
# -----------------------------
def detect_iris_and_pupil(image, model=None, model_path=default_model_path):
    """
    Uses a fine-tuned YOLO model to detect iris and pupil in the image.
    Draws bounding boxes and calculates the iris-to-pupil ratio.
    
    Args:
        image (np.array): Input image (BGR format).
        model: YOLO model instance. If None, the model is loaded using model_path.
        model_path (str): Path to the YOLO model weights.
        
    Returns:
        annotated_image (np.array): Image annotated with detection boxes.
        ratio_text (str): A text string of the calculated ratio if both iris and pupil are detected.
    """
    if model is None:
        # Directly import attempt_load from the cloned yolov5 repository.
        try:
            from yolov5.models.experimental import attempt_load
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Could not import 'yolov5.models.experimental'. Please ensure the yolov5 folder is at the root of your repository and has the correct structure."
            ) from e
        model = attempt_load(model_path, map_location="cpu")

    results = model(image)
    annotated_image = image.copy()

    iris_width = None
    pupil_width = None

    # Use the predictions tensor (results.pred). Each row in results.pred[0] is:
    # [x1, y1, x2, y2, conf, cls]
    if len(results.pred) > 0 and results.pred[0] is not None and results.pred[0].shape[0] > 0:
        preds = results.pred[0].cpu().numpy()
        for det in preds:
            x_min, y_min, x_max, y_max, conf, cls = det
            width = x_max - x_min

            if int(cls) == IRIS_CLASS:
                iris_width = width
                color = (255, 0, 0)  # Blue for iris.
            elif int(cls) == PUPIL_CLASS:
                pupil_width = width
                color = (0, 255, 0)  # Green for pupil.
            else:
                color = (0, 0, 255)  # Red for others.
            
            cv2.rectangle(annotated_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
    
    ratio_text = "Ratio: Not Detected"
    if iris_width is not None and pupil_width is not None and pupil_width != 0:
        ratio = iris_width / pupil_width
        ratio_text = f"Iris-to-Pupil Ratio: {ratio:.2f}"
    
    return annotated_image, ratio_text
