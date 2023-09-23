import streamlit as st
import tempfile
import cv2
import numpy as np
from roboflow import Roboflow
st.set_page_config(page_title="โปรแกรมนับท่อจากรูปภาพ", page_icon=":chart_with_upwards_trend:")

st.title("โปรแกรมนับท่อจากรูปภาพ V.1 (ML Version)")
rf = Roboflow(api_key="ppkxwCTvMQXUrIOdtAgF")
project = rf.workspace().project("circle-detector")
model = project.version(1).model
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
st.markdown("*หากนับท่อได้เยอะเกินจริง ให้เพิ่มค่า confidence*")
confidence=st.slider("confidence", min_value=1, max_value=100, value=40)
with st.spinner('กำลังประมวลผล . . . .'):
    if uploaded_image is not None:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_image.read())

        uploaded_image_path = tmp_file.name
        image = cv2.imread(uploaded_image_path, cv2.IMREAD_COLOR)
        # Convert BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display the uploaded image
        detections = model.predict(uploaded_image_path, confidence=confidence, overlap=30,stroke=1, labels=False)
        # Initialize the count for each class (e.g., "Pipe")
        class_counts = {}
        
        for detection in detections:
            class_name = detection["class"]
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1
            
            # Get the coordinates for centering the count text
            x_center = int(detection["x"])
            y_center = int(detection["y"])
            
            # Determine the font scale based on the object's size
            font_scale = min(detection["width"], detection["height"]) / 80.0
            
            # Calculate the size of the count text
            text_size, _ = cv2.getTextSize(str(class_counts[class_name]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            
            # Calculate the position for centering the text
            text_x = x_center - text_size[0] // 2
            text_y = y_center + text_size[1] // 2
            
            # Draw the count on the image with adjusted centering
            # Draw the white-stroked black text
            cv2.putText(image, str(class_counts[class_name]), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 10, cv2.LINE_AA)
            cv2.putText(image, str(class_counts[class_name]), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)    
        # Display the number of detected pipes and the image with counts
        if "Pipe" in class_counts:
            st.info(f"เจอท่อทั้งหมดในภาพ: {class_counts['Pipe']} ท่อน", icon="ℹ️")
        
        st.image(image, caption="", use_column_width=True)
    # Footer content
st.markdown("NPI Digital Infrastructure")
