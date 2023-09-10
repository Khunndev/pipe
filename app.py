import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
font=cv2.FONT_HERSHEY_SIMPLEX
# Function to preprocess the image and enhance pipe visibility
def preprocess_image(image):
    # Convert to grayscale
    img=cv2.medianBlur(image,3)
    plt.imshow(img)  # Convert BGR to RGB
    plt.axis('off')
    st.pyplot(plt)
    # Apply Gaussian blur to reduce noise
    #Image filtering
    blur_hor = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((11,1,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
    blur_vert = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((1,11,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
    mask = ((img[:,:,0]>blur_hor*1.2) | (img[:,:,0]>blur_vert*1.2)).astype(np.uint8)*255
    circles=cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,dp=1,minDist=60,param1=150,param2=25, minRadius=0,maxRadius=50)
    circles = np.uint16(np.around(circles))
    number_of_circles=circles.shape[1]
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,215,0),2)
    #Display the text    
    cv2.putText(img,"Total Detected pipes: "+'{}'.format(number_of_circles),(100,290),font,1,(255,255,255),2)
    plt.imshow(img[:, :, ::-1])  # Convert BGR to RGB
    plt.axis('off')
    st.pyplot(plt)
    return mask

# Function to detect and draw circles in the mask

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image_data = uploaded_image.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    plt.imshow(image[:, :, ::-1])  # Convert BGR to RGB
    plt.axis('off')
    st.pyplot(plt)
    if image is not None:
        # Preprocess the image to enhance pipe visibility
        mask = preprocess_image(image)

    else:
        st.error("Invalid image file. Please upload a valid image.")
