# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:40:59 2020

@author: Athrva Pandhare
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import statsmodels.api as sm

# Hyper parameters Here
class small_pipe_detector:
    def __init__(self, categories, minThreshold, maxThreshold,
                 filterByArea, minArea, maxArea, filterByCircularity,
                 filterByConvexity, minCircularity, minConvexity,
                 filterByInertia, minInertiaRatio):
        self.categories = categories
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.filterByArea = filterByArea
        self.minArea = minArea
        self.maxArea = maxArea
        self.filterByCircularity = filterByCircularity
        self.filterByConvexity = filterByConvexity
        self.minCircularity = minCircularity
        self.minConvexity = minConvexity
        self.filterByInertia = filterByInertia
        self.minInertiaRatio = minInertiaRatio

    
    def Initiate_detector(self):
        
        params = cv2.SimpleBlobDetector_Params() 
        params.minThreshold = self.minThreshold
        params.maxThreshold = self.maxThreshold
        # Set Area filtering parameters 
        params.filterByArea = self.filterByArea
        params.minArea = self.minArea
        params.maxArea = self.maxArea
        
        # Set Circularity filtering parameters 
        params.filterByCircularity = self.filterByCircularity
        params.minCircularity = self.minCircularity
        
        
        # Set Convexity filtering parameters 
        params.filterByConvexity = self.filterByConvexity
        params.minConvexity = self.minConvexity
        
        # Set inertia filtering parameters 
        params.filterByInertia = self.filterByInertia
        params.minInertiaRatio = self.minInertiaRatio
        
        return params
        
    def plot_diameters(self, diam):
        
        plt.title('Categorization')
        plt.xlabel('Samples')
        plt.ylabel('Diameters (in pixels)')
        plt.plot(range(len(diam)), diam)

        plt.show() 
    
    def execute(self):
        
        categories = self.categories

        image = cv2.imread('d1.png',0)
        print(image.shape)
        image = cv2.resize(image, (1920,1080), cv2.INTER_CUBIC)
        
        ret, thresh1 = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY) 
        
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(thresh1,kernel,iterations = 9)
        kernel2 = np.ones((1,1),np.uint8)
        
        opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel2)
        new_img = image + 3*opening
        thresh1 = new_img
        cv2.imshow("Filtering Circular Blobs Only", thresh1) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        
        params = self.Initiate_detector()
        
        # Create a detector with the parameters 
        detector = cv2.SimpleBlobDetector_create(params) 
        
        # Detect blobs 
        keypoints = detector.detect(thresh1) 
        diam = []
        all_pt_x = []
        all_pt_y = []
        all_pt = []
        better_keypoints = []
        if categories == 1:
            bins_num = 0
        else :
            bins_num = categories
        for kp in keypoints:
            if kp.size > 10:
                diam.append(np.round(kp.size,bins_num))
                all_pt.append(kp.pt)
                all_pt_x.append(kp.pt[0])
                all_pt_y.append(kp.pt[1])
        valid_data = list(np.histogram(diam,bins = 10))
        

        

        if categories == 1:
            best_index_search = list(valid_data[0]).index(max(valid_data[0]))
            valid_data = [max(valid_data[0][:]), valid_data[1][best_index_search : best_index_search + 2]]
            diff = np.diff(valid_data[1])
            # Selecting better keypoints here based on the frequency of detections
            for kp2 in keypoints: 
                k = max(valid_data[1])
                
                if  k - (np.round(kp2.size, bins_num)) > 0 and k - (np.round(kp2.size, bins_num))  < diff:
                    better_keypoints.append(kp2)
        else:
            print(valid_data)
            temp_data_ind = []
            temp_data_range = []
            best_index_search = []
            valid_data[0] = list(valid_data[0])
            buffer = sorted(valid_data[0], reverse = True)
            for i in buffer[0:categories]:
                best_index_search.append(valid_data[0].index(i))
            for index in best_index_search:
                temp_data_ind.append(valid_data[0][index])
                temp_data_range.append(valid_data[1][index : index + 2])
            valid_data = [(temp_data_ind), (temp_data_range)]
            for kp2 in keypoints: 
                for k in  temp_data_range:
                    m  = max(k)
                    diff = np.diff(k)
                    if  m - (np.round(kp2.size, bins_num)) > 0 and m - (np.round(kp2.size, bins_num))  < diff:
                        better_keypoints.append(kp2)
                     
            print("Valid data is : ", valid_data)
        


        # Draw blobs on our image as red circles 
        blank = np.zeros((1, 1))  
        

        blobs = cv2.drawKeypoints(image, better_keypoints, blank, (0, 0, 255), 
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
        
        number_of_blobs = len(better_keypoints) 
        text = "Number of Objects: " + str(number_of_blobs) 
        cv2.putText(blobs, text, (20, 550), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
        
        # Show blobs 
        cv2.imshow("Filtering Circular Blobs Only", blobs) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        print("Number of Circles Detected = ", number_of_blobs)
        
        self.plot_diameters(diam)

def main():
    categories = 4 # Number of different scales of the objects.
    minThreshold = 0
    maxThreshold = 255
    filterByArea = True
    minArea = 100
    maxArea = 10000000       
    filterByCircularity = True
    minCircularity = 0.5
    filterByConvexity = True
    minConvexity = 0.19
    filterByInertia = True
    minInertiaRatio = 0.01
    spd = small_pipe_detector(categories, minThreshold,
                              maxThreshold, filterByArea,
                              minArea, maxArea,
                              filterByCircularity, filterByConvexity,
                              minCircularity, minConvexity,
                              filterByInertia, minInertiaRatio)
    spd.execute()

if __name__ == '__main__':
    main()