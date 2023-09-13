import streamlit as st
import cv2
import numpy as np
import tempfile
st.title("Blob Detection App")

# Upload an image
image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image is not None:
    # Load the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(image.read())

    uploaded_image_path = tmp_file.name
    image = cv2.imread(uploaded_image_path)
    # Preprocess the image (you can adjust these parameters as needed)
    gray = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    st.image(thresh, caption="thresh", use_column_width=True)
    # Create blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect circular blobs on the preprocessed image
    keypoints = detector.detect(thresh)

    # Create a copy of the image to draw circles and numbers on
    result_image = image.copy()

    # Define a target font size relative to the circle's radius (adjust as needed)
    target_font_size = 0.05
    # Add numbers to the center of each circle with a font size relative to the circle's radius
    for i, keypoint in enumerate(keypoints, start=1):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        radius = int(keypoint.size / 2.4)

        # Draw the circle
        cv2.circle(result_image, (x, y), radius, (0, 0, 255), 2)

        # Add the number at the center of the circle with the calculated font size
        cv2.putText(result_image, str(i), (x - radius, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Get the total number of circular blobs
    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(number_of_blobs)

    st.write(text)

    # Display the result using Streamlit
    st.image(result_image, channels="BGR", caption="Blob Detection Result", use_column_width=True)

    # Add input field to remove keypoints by number
    st.header("Remove Keypoints by Number")
    remove_numbers = st.text_input("Enter the numbers of keypoints to remove (comma-separated):")

    if remove_numbers:
        remove_numbers = [int(num.strip()) for num in remove_numbers.split(",")]
        keypoints = [keypoint for i, keypoint in enumerate(keypoints, start=1) if i not in remove_numbers]

        # Create a new result image without removed keypoints
        result_image = image.copy()

        for i, keypoint in enumerate(keypoints, start=1):
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            radius = int(keypoint.size / 2)

            # Draw the circle
            cv2.circle(result_image, (x, y), radius, (0, 0, 255), 2)

            # Add the number inside the circle with the calculated font size
        cv2.putText(result_image, str(i), (x - radius, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Display the updated result image
        st.image(result_image, channels="BGR", caption="Updated Blob Detection Result", use_column_width=True)