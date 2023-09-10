import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import datetime

st.title("PVC Pipe Circle Drawer with Counts (No Center Dot)")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_image.read())

    uploaded_image_path = tmp_file.name

    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    img = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    
    st.image(img, caption="equalizeHist", use_column_width=True)
    img = cv2.bitwise_not(img)
    st.image(img, caption="bitwise_not", use_column_width=True)
    auto_radius = int(5)  # Adjust this factor as needed

    # Slider for adjusting radius
    radius_slider = st.slider("Radius", min_value=1, max_value=100, value=auto_radius)

    # Button to find circles
    if True:
        rmin = radius_slider
        rmax = 3 * rmin

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=rmin, maxRadius=rmax)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            num_circles = len(circles[0, :])
            st.write(f"Number of Circles: {num_circles}")

            # Create an image with a transparent white background
            img_with_circles = cv2.imread(uploaded_image_path)
            overlay = img_with_circles.copy()
            alpha = 0.5  # Transparency level

            for idx, circle in enumerate(circles[0, :], start=1):
                x, y, r = circle

                # Draw a red border around the circle
                cv2.circle(overlay, (x, y), r, (255, 0, 0), 2)

                # Draw a white filled circle (transparent)
                cv2.circle(overlay, (x, y), r - 2, (255, 255, 255), -1)

                # Add the circle number at the center with reduced font size
                font_scale = 0.5
                font_thickness = 1
                text_size, _ = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(overlay, str(idx), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            # Combine the overlay with the original image
            img_with_circles = cv2.addWeighted(overlay, alpha, img_with_circles, 1 - alpha, 0)
            # Add text for the total number of circles and date/time with a gray background
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = f"Total Circles: {num_circles} | Timestamp: {timestamp}"
            cv2.putText(img_with_circles, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # Display the image with circles and numbers
            st.image(img_with_circles, caption="Image with Circles", use_column_width=True)
        else:
            st.info("No circles found.")
