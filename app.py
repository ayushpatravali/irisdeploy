import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from irispupil import detect_face_and_eyes, enhance_image, detect_iris_and_pupil

# Define directories (should match those used in irispupil.py)
detected_eyes_dir = "detected_eyes"
enhanced_eyes_dir = "enhanced_eyes"
processed_eyes_dir = "processed_eyes"
os.makedirs(detected_eyes_dir, exist_ok=True)
os.makedirs(enhanced_eyes_dir, exist_ok=True)
os.makedirs(processed_eyes_dir, exist_ok=True)

# Configure the Streamlit page.
st.set_page_config(page_title="Iris & Pupil Detection", layout="wide")
st.title("Iris & Pupil Detection App")

# Sidebar options.
st.sidebar.header("Options")
save_detect = st.sidebar.checkbox("Save Detected Eyes", value=True)
input_mode = st.sidebar.radio("Select Input Mode", ("Camera", "Upload Raw Image", "Upload Enhanced Image"))

# Get image input.
if input_mode == "Camera":
    img_file = st.camera_input("Capture an image")
else:
    # For both raw and enhanced images, use file uploader.
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

    if input_mode in ["Camera", "Upload Raw Image"]:
        # --- Step 1: Eye Detection ---
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
            enhanced_results = []
            # Process each saved eye image in the detected_eyes directory.
            for filename in os.listdir(detected_eyes_dir):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    detected_path = os.path.join(detected_eyes_dir, filename)
                    # Read the cropped eye image.
                    eye_img = cv2.imread(detected_path)
                    # Enhance the image.
                    enhanced = enhance_image(eye_img)
                    # Save the enhanced image.
                    enhanced_path = os.path.join(enhanced_eyes_dir, filename)
                    cv2.imwrite(enhanced_path, enhanced)
                    enhanced_results.append((filename, enhanced))
            if enhanced_results:
                st.write("Enhanced Images:")
                for fname, enh_img in enhanced_results:
                    st.image(enh_img, caption=f"Enhanced: {fname}", use_column_width=True)
            else:
                st.info("No detected eye images found for enhancement.")
            
            # --- Step 3: Iris & Pupil Detection on Enhanced Eyes ---
            st.subheader("Iris & Pupil Detection on Enhanced Eyes")
            detection_results = []
            for filename in os.listdir(enhanced_eyes_dir):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    enhanced_path = os.path.join(enhanced_eyes_dir, filename)
                    # Read the enhanced image.
                    enh_img = cv2.imread(enhanced_path)
                    detection_img, ratio_text = detect_iris_and_pupil(enh_img)
                    detection_results.append((filename, detection_img, ratio_text))
            if detection_results:
                for fname, det_img, ratio in detection_results:
                    st.image(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB),
                             caption=f"Detection on {fname}: {ratio}",
                             use_column_width=True)
            else:
                st.info("No enhanced images found for iris & pupil detection.")
        else:
            st.info("Detected eyes were not saved. Enable 'Save Detected Eyes' in the sidebar to run the full pipeline.")
    
    elif input_mode == "Upload Enhanced Image":
        # For enhanced images, we skip detection & enhancement steps.
        st.subheader("Using Uploaded Enhanced Image for Detection")
        # Save the uploaded image in the enhanced directory.
        filename = img_file.name
        enhanced_path = os.path.join(enhanced_eyes_dir, filename)
        cv2.imwrite(enhanced_path, img)
        # Run detection directly.
        detection_img, ratio_text = detect_iris_and_pupil(img)
        # Save the processed image in the processed directory.
        processed_path = os.path.join(processed_eyes_dir, filename)
        cv2.imwrite(processed_path, detection_img)
        st.image(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB),
                 caption=f"Detection on {filename}: {ratio_text}",
                 use_column_width=True)
    
    st.sidebar.success("Processing complete!")
