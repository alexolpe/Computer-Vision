import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from scipy.ndimage import maximum_filter

## HARRIS APPROACH---------------------------------------------------
# Function to draw points
def putPointsOnImage(corners, image):
    for corner in corners:
        cv2.circle(image, tuple(corner), radius=5, color=(42, 247, 237), thickness =-1)
    return image
    
# Function to draw points and lines
def putPointsAndLinesOnImage(final_image, w, selected, p1, corners2):
    points_1=(p1)
    points_2 = (corners2[selected] + np.array([w, 0]))
    cv2.circle(final_image, points_1, radius=5, color = (42, 247, 237), thickness=-1)
    cv2.circle(final_image, points_2, radius = 5, color=(42, 247, 237), thickness =-1)
    cv2.line(final_image,points_1,points_2,(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)
    return final_image

# Function to get the Harris response
def getHarrisResponse(dx, dy, N):
    s_dxdx = cv2.filter2D(dx*dx, ddepth=-1, kernel=np.ones((N,N)))
    s_dydy = cv2.filter2D(dy*dy, ddepth=-1, kernel=np.ones((N,N)))
    s_dxdy = cv2.filter2D(dx*dy, ddepth=-1, kernel=np.ones((N,N)))
    #Compute the trace and determinant
    tr_c = s_dxdx + s_dydy
    det_c = (s_dxdx*s_dydy)-(s_dxdy*s_dxdy)
    return det_c-0.05*(tr_c**2)
    
def harris_detect_corner(image, sig):
    #Get gray scale image
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
    
    # Build Haar filter
    M = int(np.ceil(4*sig))
    if M%2 == 1:
        M = M + 1
    haar_x = np.zeros((M, M))
    haar_x[:, :M // 2] = -1
    haar_x[:, M // 2:] = 1
    haar_y = np.zeros((M, M))
    haar_y[:M // 2, :] = 1
    haar_y[M // 2:, :] = -1
    
    # Get dx and dy by applying Haar filter in both directions
    dx = cv2.filter2D(gray_scale_image, ddepth = -1, kernel = haar_x)
    dy = cv2.filter2D(gray_scale_image, ddepth = -1, kernel = haar_y)
    
    #Get Harris response and set threshold
    N = int(np.ceil(5*sig))
    if N%2 == 1:
        N = N + 1
    H = getHarrisResponse(dx, dy, N)
    R_abs = np.abs(H)
    thr = np.mean(R_abs)
    K = 2*N
    
    #Apply maximum filter to get local max values in (2K+1, 2K+1) windows
    R_max_window = maximum_filter(H, size=(2*K+1, 2*K+1))
    #Create a boolean mask where the current pixel is the max in its window and exceeds threshold
    max_mask = (H == R_max_window) & (H > thr)
    #Find the coordinates of the corners
    y_coords, x_coords = np.nonzero(max_mask)
    #Extract the corresponding R values at those positions
    R_values = H[y_coords, x_coords]
    #Stack coordinates with their corresponding R values
    corners = np.vstack((x_coords, y_coords, R_values)).T
    #Sort corners based on the R value (descending) and take the top 100
    corners_100 = corners[np.argsort(corners[:, 2])][-100:, :2].astype(int)

    img_with_corners = putPointsOnImage(corners_100, image)
        
    return [corners_100, img_with_corners]

def ssd(image1, image2, fake_corners_1, fake_corners_2, window):
    #Get gray scale image
    gray_scale_image_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)/255
    gray_scale_image_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)/255
    final_image = np.concatenate((image1, image2), axis=1)
       
    # We remove the pixels that do not have enough margin to get the window of pixels around them
    corners_1 = fake_corners_1[
        (fake_corners_1[:, 1] - window // 2 > 0) &
        (fake_corners_1[:, 0] - window // 2 > 0) &
        (fake_corners_1[:, 1] + window // 2 < gray_scale_image_1.shape[0]) &
        (fake_corners_1[:, 0] + window // 2 < gray_scale_image_1.shape[1])
    ]
    corners_2 = fake_corners_2[
        (fake_corners_2[:, 1] - window // 2 > 0) &
        (fake_corners_2[:, 0] - window // 2 > 0) &
        (fake_corners_2[:, 1] + window // 2 < gray_scale_image_2.shape[0]) &
        (fake_corners_2[:, 0] + window // 2 < gray_scale_image_2.shape[1])
    ]
    
    #Match corners similar to previous solution
    for p1 in corners_1:
        ssd = np.zeros((len(corners_2),2))
        for j , p2 in enumerate ( corners_2 ):
            image1_window = gray_scale_image_1[p1[1]-window//2:p1[1]+window//2, p1[0]-window//2:p1[0]+window//2]
            image2_window = gray_scale_image_2[p2[1]-window//2:p2[1]+window//2, p2[0]-window//2:p2[0]+window//2]
            ssd[j] = np.array([np.sum((image1_window-image2_window)**2),j])
        final_image = putPointsAndLinesOnImage(final_image, image1.shape[1], np.argmin(ssd[:,0], axis=0), p1, corners_2)
    return final_image

def ncc(image1, image2, fake_corners_1, fake_corners_2, window):
    #Get gray scale image
    gray_scale_image_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)/255
    gray_scale_image_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)/255
    final_image = np.concatenate((image1, image2), axis=1)
    
    # We remove the pixels that do not have enough margin to get the window of pixels around them
    corners_1 = fake_corners_1[
        (fake_corners_1[:, 1] - window // 2 > 0) &
        (fake_corners_1[:, 0] - window // 2 > 0) &
        (fake_corners_1[:, 1] + window // 2 < gray_scale_image_1.shape[0]) &
        (fake_corners_1[:, 0] + window // 2 < gray_scale_image_1.shape[1])
    ]
    corners_2 = fake_corners_2[
        (fake_corners_2[:, 1] - window // 2 > 0) &
        (fake_corners_2[:, 0] - window // 2 > 0) &
        (fake_corners_2[:, 1] + window // 2 < gray_scale_image_2.shape[0]) &
        (fake_corners_2[:, 0] + window // 2 < gray_scale_image_2.shape[1])
    ]
    
    #Match corners similar to previous solution
    for p1 in corners_1:
        ncc = np.zeros((len(corners_2),2))
        for j, p2 in enumerate(corners_2):
            image1_window = gray_scale_image_1[p1[1]-window//2:p1[1]+window//2, p1[0]-window//2:p1[0]+window//2]
            image2_window = gray_scale_image_2[p2[1]-window//2:p2[1]+window//2, p2[0]-window//2:p2[0]+window//2]
            ncc[j] = np.array([np.sum((image1_window-np.mean(image1_window))*(image2_window-np.mean(image2_window)))/np.sqrt((np.sum((image1_window-np.mean(image1_window))**2))*(np.sum((image2_window-np.mean(image2_window))** 2))),j])
        if np.max(ncc[:,0])>0.3:
            selected_ncc = np.argmax(ncc[:,0], axis=0)
        else: break
        final_image = putPointsAndLinesOnImage(final_image, image1.shape[1], selected_ncc, p1, corners_2)
    return final_image

#Example for the hovde image. We would just change the path for the other images
image1 = cv2.imread("/home/aolivepe/Computer-Vision/HW4/HW4_images/hovde_1.jpg")
image2 = cv2.imread("/home/aolivepe/Computer-Vision/HW4/HW4_images/hovde_2.jpg")

for sig in [0.8, 1.2, 1.6, 2.0]:
    #Get corners from first image
    corners1, h_image1 = harris_detect_corner(image1, sig)
    cv2.imwrite(f'./output/corners_1_{str(sig)}.jpeg', h_image1)
    
    #Get corners from second image
    corners2, h_image2 = harris_detect_corner(image2, sig)
    cv2.imwrite(f'./output/corners_2_{str(sig)}.jpeg', h_image2)

    #Match corners using ssd
    ssd_final = ssd(image1, image2, corners1, corners2, 10)
    cv2.imwrite(f'./output/ssd_{str(sig)}.jpeg', ssd_final)

    #Match corners using ncc
    ncc_final = ncc(image1, image2, corners1, corners2, 10)
    cv2.imwrite(f'./output/ncc_{str(sig)}.jpeg', ncc_final)

## SIFT APPROACH -----------------------------------------------------------------------------------------
#Example for the hovde image. We would just change the path for the other images
image1 = cv2.imread("/home/aolivepe/Computer-Vision/HW4/HW4_images/hovde_1.jpg")
image2 = cv2.imread("/home/aolivepe/Computer-Vision/HW4/HW4_images/hovde_2.jpg")

# Create a SIFT detector instance
sift_detector = cv2.SIFT_create()

# Extract keypoints and vectors from both images
kp1, vector1 = sift_detector.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
kp2, vector2 = sift_detector.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)

# Setup the Brute Force Matcher
raw_matches = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True).match(vector1, vector2)

# Sort matches by distance
sorted_matches = []
for match in raw_matches:
    sorted_matches.append(match)
sorted_matches.sort(key=lambda m: m.distance)

# Draw the top 100 matches on the images
final_image = cv2.drawMatches(image1, kp1, image2, kp2, sorted_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Construct the output file name and save the image
cv2.imwrite("./output/sift.jpeg", final_image)
