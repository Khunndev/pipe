import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to count circles of a specific color and return their positions
def count_circles(image, color):
    lower = np.array(color[0:3], dtype="uint8")
    upper = np.array(color[3:6], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    # Apply Gaussian blur to reduce noise and improve accuracy
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]
    else:
        return []

# Function to draw circles on an image with counts and no center dot
def draw_circles_with_counts(image, circles):
    image_with_circles = image.copy()
    for i, circle in enumerate(circles):
        center = (circle[0], circle[1])
        radius = circle[2]
        # Draw blue circle border
        cv2.circle(image_with_circles, center, radius, (255, 0, 0), 2)

        # Draw white filled circle
        cv2.circle(image_with_circles, center, radius, (255, 255, 255), -1)

        # Draw count text inside the circle
        count_text = str(i + 1)
        text_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = circle[0] - text_size[0] // 2
        text_y = circle[1] + text_size[1] // 2
        cv2.putText(image_with_circles, count_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image_with_circles
# Streamlit app
def main():
    st.title("PVC Pipe Circle Drawer with Counts (No Center Dot)")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image_data = uploaded_image.read()  # Read image data
        image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)  # Convert to NumPy array

        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            blue_circles = count_circles(image.copy(), [100, 100, 0, 255, 255, 100])
            gray_circles = count_circles(image.copy(), [150, 150, 150, 255, 255, 255])
            yellow_circles = count_circles(image.copy(), [0, 100, 100, 100, 255, 255])

            st.write(f"Blue Circles: {len(blue_circles)}")
            st.write(f"Gray Circles: {len(gray_circles)}")
            st.write(f"Yellow Circles: {len(yellow_circles)}")

            # Draw circles with counts (no center dot) on the image
            image_with_circles = image.copy()
            image_with_circles = draw_circles_with_counts(image_with_circles, blue_circles)
            image_with_circles = draw_circles_with_counts(image_with_circles, gray_circles)
            image_with_circles = draw_circles_with_counts(image_with_circles, yellow_circles)

            # Plot the image with circles and counts
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.error("Invalid image file. Please upload a valid image.")

if __name__ == "__main__":
    main()
