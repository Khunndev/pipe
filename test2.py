# import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

#load image
image=cv2.imread("d2.png")
#cv2.imshow("figure",image)
cv2.waitKey(0)
#create blob detector
detector=cv2.SimpleBlobDetector_create()
#keypoints
keypoints=detector.detect(image)
#create a blank file which is a numpy array to hold our circle that we will detect and mark during the process
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobss: " + str(len(keypoints))
cv2.putText(blobs, text, (2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

# Display image with blob keypoints
#cv2.imshow("Blobs using default parameters", blobs)
cv2.waitKey(0)
#another function to create blobs
params=cv2.SimpleBlobDetector_Params()
#AREA
params.filterByArea=True
params.minArea=100
#CIRCULARITY
params.filterByCircularity=True
params.minCircularity=0.4
#CONVEXITY
params.filterByConvexity=True
params.minConvexity=0.3
#INERTIA
params.filterByInertia=True
params.minInertiaRatio=0.01

detector=cv2.SimpleBlobDetector_create(params)
keypoints=detector.detect(image)
blank=np.zeros((1,1))
number_of_blobs = len(keypoints)

#for i in range(number_of_blobs):
print("Number of Circular Blobs: " + str(number_of_blobs))

blobs=cv2.drawKeypoints(image,keypoints,blank,(0,200,222),cv2.DRAW_MATCHES_FLAGS_DEFAULT)

#number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))

cv2.putText(blobs, text, (1, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


cv2.imshow("blobs",blobs)
cv2.waitKey(0)



cv2.destroyAllWindows()