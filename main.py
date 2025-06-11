import streamlit as st
from PIL import Image
from ultralytics import YOLO
from scripts.model import predict_yolo_single_image
import os
import warnings
warnings.filterwarnings("ignore")



# Streamlit app for detecting Bangkok Metro Station Signage using YOLOv8

# Load YOLOv8 model with transfer learning
@st.cache_resource
def load_yolo_model():
    '''Load the YOLOv8 model trained for Bangkok Metro Station Signage detection.'''
    best_yolo_model_path = os.path.join("models", "deep_learning", "best.pt")
    model = YOLO(best_yolo_model_path)
    return model



def run():
    '''Run the Streamlit app for detecting Bangkok Metro Station Signage.'''

    # Streamlit page title
    st.title("Bangkok Metro Station Signage Detection")
    st.markdown('**This is a demo application for identifying metro station name and signage from images containing metro station signage, for 8 skytrain stations in BTS Silom Line (Dark Green Line without extension:).**')
    hide_footer_style = """<style>.reportview-container .main footer {visibility: hidden;}"""
    st.markdown(hide_footer_style, unsafe_allow_html=True)

    # Load the YOLOv8 model
    model = load_yolo_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload an image with BTS Silom Line station signage", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.subheader("Uploaded Image")
        # Open uploaded_image and convert it to RGB
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Predict metro station and draw bounding boxes in the uploaded image
        station_names_unique, new_resulting_image_path = predict_yolo_single_image(model, uploaded_image)

        # Display the detection result
        st.subheader("Detection Result")
        st.write("Detected Station:")
        if station_names_unique:
            for i, name in enumerate(station_names_unique, 1):
                st.write(f"{i}. {name}")
        else:
            st.write("No station detected.")

        # Display the resulting image with bounding boxes
        st.image(new_resulting_image_path, caption="Detected Metro Station Signage", use_container_width=True)

    else:
        st.warning("Please upload an image file to proceed.")



if __name__ == "__main__":
    run()
