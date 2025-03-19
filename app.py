import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from irispupil import detect_face_and_eyes, enhance_image, detect_iris_and_pupil

# Define directories (should match what your functions use)
detected_eyes_dir = "detected_eyes"
enhanced_eyes_dir = "enhanced_eyes"
os.makedirs(detected_eyes_dir, exist_ok=True)
os.makedirs(enhanced_eyes_dir, exist_ok=True)

# Configure the Streamlit page.
st.set_page_config(page_title="Iris & Pupil Detection", layout="wide")
st.title("Iris & Pupil Detection App")

# Sidebar options.
st.sidebar.header("Options")
save_detect = st.sidebar.checkbox("Save Detected Eyes", value=True)
input_mode = st.sidebar.radio("Select Input Mode", ("Camera", "Upload Image"))

# Get image input.
if input_mode == "Camera":
    img_file = st.camera_input("Capture an image")
else:
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    # Read image from file.
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Initialize or update person counter.
    if "person_counter" not in st.session_state:
        st.session_state.person_counter = 1

    # --- Step 1: Eye Detection ---
    # This function detects eyes, annotates the image, and (if enabled) saves cropped eye images.
    annotated_img, eyes = detect_face_and_eyes(
        img, save_detected=save_detect, person_counter=st.session_state.person_counter
    )
    st.subheader("Eye Detection")
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    if save_detect:
        st.success(f"Detected eyes saved for person {st.session_state.person_counter}.")
        st.session_state.person_counter += 1
        
        # --- Step 2: Image Enhancement on Detected Eyes ---
        st.subheader("Enhancing Detected Eyes")
        enhanced_results = []  # To store tuples of (filename, enhanced image)
        
        # Process each image in the detected eyes directory.
        for filename in os.listdir(detected_eyes_dir):
            detected_path = os.path.join(detected_eyes_dir, filename)
            # Check for valid image file.
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Read the cropped eye image.
                eye_img = cv2.imread(detected_path)
                # Enhance the image (this returns a grayscale enhanced image).
                enhanced = enhance_image(eye_img)
                # Save enhanced image.
                enhanced_path = os.path.join(enhanced_eyes_dir, filename)
                cv2.imwrite(enhanced_path, enhanced)
                enhanced_results.append((filename, enhanced))
        
        if enhanced_results:
            st.write("Enhanced images:")
            for fname, enh_img in enhanced_results:
                st.image(enh_img, caption=f"Enhanced: {fname}", use_column_width=True)
        else:
            st.info("No detected eye images found to enhance.")
        
        # --- Step 3: Iris & Pupil Detection on Enhanced Images ---
        st.subheader("Iris & Pupil Detection on Enhanced Eyes")
        detection_results = []
        for filename in os.listdir(enhanced_eyes_dir):
            enhanced_path = os.path.join(enhanced_eyes_dir, filename)
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Read the enhanced image.
                enh_img = cv2.imread(enhanced_path)
                # Run iris & pupil detection using the YOLO model.
                detection_img, ratio_text = detect_iris_and_pupil(enh_img)
                # Optionally, you can save the detection image back to a folder (or display it).
                detection_results.append((filename, detection_img, ratio_text))
        
        if detection_results:
            for fname, det_img, ratio in detection_results:
                st.image(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB),
                         caption=f"Detection on {fname}: {ratio}",
                         use_column_width=True)
        else:
            st.info("No enhanced images found for iris & pupil detection.")
        
    else:
        st.info("Detected eyes were not saved. To run full pipeline (detection → enhancement → evaluation), enable 'Save Detected Eyes' in the sidebar.")

    st.sidebar.success("Processing complete!")
