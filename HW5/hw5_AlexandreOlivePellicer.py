import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from scipy.ndimage import maximum_filter
import math
from einops import rearrange
from tqdm import trange , tqdm
import os
from scipy.optimize import least_squares, approx_fprime

np.random.seed(42) 
    
def ImageStitch(images, homographies):
    # homographies:
    # [0] img1->img2
    # [1] img2->img3
    # [2] img3->img4
    # [3] img4->img5
    
    center_index = len(images) // 2
    updated_homographies = [None] * (len(homographies) + 1)
    
    # Initialize the transformation for the center image
    updated_homographies[center_index] = np.eye(3)
    
    # Calculate transformations to the left of the center image
    for i in range(center_index - 1, -1, -1):
        updated_homographies[i] = updated_homographies[i + 1] @ homographies[i]
    
    # Calculate transformations to the right of the center image
    for i in range(center_index, len(homographies)):
        updated_homographies[i + 1] = updated_homographies[i] @ np.linalg.inv(homographies[i])

    homographies = updated_homographies 

    # Find the final dimensions of the image following the method from TA in Piazza
    full_corners = []
    # Get corners of each image
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]   
        corners = np.array([[0, 0, 1],
                            [w, 0, 1],	
                            [w, h, 1],
                            [0, h, 1]
                            ])
        corners_hc = corners.T
        corners_transformed_hc = np.matmul(H, corners_hc)
        corners_homogeneous = corners_transformed_hc / (corners_transformed_hc[-1, :])
        full_corners.append(corners_homogeneous.T)
    full_corners = np.concatenate(full_corners, axis=0)
    
    # Find the maximum and minimums corners
    x_min = int(min(full_corners[:, 0]))
    x_max = int(max(full_corners[:, 0]))
    y_min = int(min(full_corners[:, 1]))
    y_max = int(max(full_corners[:, 1]))

    # Final dimensions of the image
    final_size = (y_max - y_min, x_max - x_min)
   
    # Translation to put all the images inside the found limits
    trans = np.array([[1, 0, -x_min],
                    [0, 1, -y_min],
                    [0, 0, 1]])

    # Adjust homographies by applying the offset translation
    adjusted_homographies = [trans @ h for h in homographies]
    
    # Initialize panoramic and mask accumulator for blending
    panoramic = np.zeros((final_size[0], final_size[1], 3), dtype=np.float32)
    full_mask = np.zeros(final_size, dtype=np.float32)

    for img, h in zip(images, adjusted_homographies):
        # Apply the adjusted homography to each image
        warped = cv2.warpPerspective(img, h, final_size[::-1])
        
        # Create mask to identify visible areas in the warped image
        mask = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        
        # Blend each warped image onto the panoramic
        panoramic += warped * np.expand_dims(mask, axis=2)
        full_mask += mask

    # Normalize the panoramic to avoid bright spots
    panoramic /= (np.expand_dims(full_mask, axis=2) + 1e-8)
    panoramic = np.clip(panoramic, 0, 255).astype(np.uint8)

    return panoramic

# Method to get homography h by least squares method
def obtain_h(vector_x, vector_x_prima):
    # Build matrix A and vector b
    A = []
    b = []
    for x, x_prima in zip(vector_x, vector_x_prima):
        A += [[0, 0, 0, -x_prima[2]*x[0], -x_prima[2]*x[1], -x_prima[2]*x[2], x_prima[1]*x[0], x_prima[1]*x[1]],
                [x_prima[2]*x[0], x_prima[2]*x[1], x_prima[2]*x[2], 0, 0, 0, -x_prima[0]*x[0], -x_prima[0]*x[1]]]

        b += [[-x_prima[1]*x[2]], 
                [x_prima[0]*x[2]]]

    # Compute homography and reshape to 3x3
    A = np.array(A)
    b = np.array(b)
    h = np.linalg.pinv(A.T @ A) @ A.T @ b
    h = np.append(h, 1)
    h = h.reshape(3, 3)
    return h

def get_inliers(hc_kp1, hc_kp2, h, delta):
    # Get projected points in img1 to img2 (estimations)
    projected_hc_kp2 = h @ hc_kp1.T
    projected_kp2 = projected_hc_kp2/projected_hc_kp2[2]
    # Compute distance between actual point and estimated one
    dist_x_y = (hc_kp2[:, :2] - projected_kp2.T[:, :2])**2
    dist = np.sum(dist_x_y, axis=1)
    # Inlier points are the points with distance below delta
    inliers_pos =  np.where(dist <= delta)
    return inliers_pos[0]
  
def ransac(hc_kp1, hc_kp2, n = 20, sigma = 2, p = 0.99, e = 0.4):
    # increase n to increase N
    # increase p to increase N
    # increase e to increase N and decrease M
    # Set parameters of the ransac algorithm
    delta = 3* sigma
    N = int(np.ceil(np.log(1-p)/np.log(1-(1-e)**n)))
    print("N: ", N)
    n_total = len(hc_kp1)
    M = (1-e)*n_total
    previous_len = 0
    
    # Iterate over N
    for i in range(N):
        # Select n random keypoints and compute homography with them
        random_idx = np.random.randint(n_total, size = n)
        random_kp1 = hc_kp1[random_idx]
        random_kp2 = hc_kp2[random_idx]
        h = obtain_h(random_kp1, random_kp2)
        
        # Use the computed homography to determine the inlier set
        inliers_pos = get_inliers(hc_kp1, hc_kp2, h, delta)
        
        # Save inlier set if it was bigger than previous one and stop iterating if inlier set is bigger than M
        if len(inliers_pos) > previous_len:
            previous_len = len(inliers_pos)
            final_inliers_pos = inliers_pos
            if len(inliers_pos) > M:
                break           
    print("iterartion: ", i, "Num inliers: ", len(final_inliers_pos))
    return final_inliers_pos

def putPointsAndLines(img1, img2, hc_inliers_kp1, hc_inliers_kp2, hc_outliers_kp1, hc_outliers_kp2):
    # Paint keypoints and matches in the concatenation of the 2 images
    concatenated_image = np.hstack((img1, img2))
    img1_width = img1.shape[1]
    
    # Paint outliers
    for pt1, pt2 in zip(hc_outliers_kp1, hc_outliers_kp2):
        cv2.circle(concatenated_image, (int(pt1[0]), int(pt1[1])), radius=3, color=(0, 0, 255), thickness=-1)
        pt2_adjusted = (int(pt2[0] + img1_width), int(pt2[1]))
        cv2.circle(concatenated_image, pt2_adjusted, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.line(concatenated_image, (int(pt1[0]), int(pt1[1])), pt2_adjusted, color=(0, 0, 255), thickness=1)
        
    # Paint inliers
    for pt1, pt2 in zip(hc_inliers_kp1, hc_inliers_kp2):
        cv2.circle(concatenated_image, (int(pt1[0]), int(pt1[1])), radius=3, color=(0, 255, 0), thickness=-1)
        pt2_adjusted = (int(pt2[0] + img1_width), int(pt2[1]))
        cv2.circle(concatenated_image, pt2_adjusted, radius=3, color=(0, 255, 0), thickness=-1)
        cv2.line(concatenated_image, (int(pt1[0]), int(pt1[1])), pt2_adjusted, color=(0, 255, 0), thickness=1)
    return concatenated_image

def error_function(p, kp1_hc, kp2_hc):
    h = p.reshape(3, 3)
    # Get projected points in img1 to img2 (estimations)
    project_kp2_hc = h @ kp1_hc.T
    project_kp2_hc = project_kp2_hc / project_kp2_hc[2]
    project_kp2_hc = project_kp2_hc.T
    # Get only (x, y) coordinates of the estimation and the actual points
    kp2 = kp2_hc[:, :-1]
    project_kp2 = project_kp2_hc[:, :-1]
    # Compute error between the estimation and the actual points
    X = kp2.reshape(kp2.shape[0]*2)
    f = project_kp2.reshape(project_kp2.shape[0]*2)
    e = X - f
    return e

def jacobian(kp1_hc, h):
    num_points = len(kp1_hc)
    J = np.zeros((num_points * 2, 9))  # Initialize the Jacobian matrix

    for i, pt in enumerate(kp1_hc):        
        # Apply homography transformation
        transformed_pt = h @ pt  
        
        # Build Jacobian matrix
        x, y, _ = pt
        fx, fy, fw = transformed_pt
        inv_fw = 1 / fw
        inv_fw2 = inv_fw ** 2
        J[2 * i, :3] = np.array([x * inv_fw, y * inv_fw, inv_fw])
        J[2 * i, 6:] = np.array([-x * fx * inv_fw2, -y * fx * inv_fw2, -fx * inv_fw2])
        J[2 * i + 1, 3:6] = np.array([x * inv_fw, y * inv_fw, inv_fw])
        J[2 * i + 1, 6:] = np.array([-x * fy * inv_fw2, -y * fy * inv_fw2, -fy * inv_fw2])

    return J

def levemberg_marquardt(p, hc_kp1, hc_kp2, r = 0.5, K_total = 100, th = 1e-20):
    # r = 0.5, K_total = 50000, th = 1e-13
    # Build Jacobian
    J = jacobian(hc_kp1, p.reshape(3, 3))
    # Estimate initial mu
    mu = r*np.max(np.diagonal(J.T @ J))
    # Vector to save how the geometric error is reduced
    C_p_vector = []
    # Refine for 100 iterations
    for i in range(K_total):
        # Get error
        e = error_function(p, hc_kp1, hc_kp2)
        # Get cost
        C_p = np.linalg.norm(e)**2
        # Update Jacobian with new p
        J = jacobian(hc_kp1, p.reshape(3, 3))
        # Get delta_p
        delta_p = np.linalg.inv(J.T @ J + mu * np.eye(9)) @ J.T @ e  
        # Update p    
        p = p + delta_p        
        # Compute phi
        C_p_plus_1 = np.linalg.norm(error_function(p, hc_kp1, hc_kp2))**2
        phi = (C_p - C_p_plus_1) / (delta_p.T @ J.T @ e + delta_p.T @ (mu * np.eye(9)) @ delta_p)
        # Update mu
        mu = mu * max(1/3, 1-(2*phi-1)**3)
        # Append geometric error
        C_p_vector.append(C_p_plus_1)
    return p, C_p_vector

if __name__ == '__main__':
    # Read images
    img1 = cv2.imread(f"/home/aolivepe/HW5/my_HW5_images_2/1.jpg")
    img2 = cv2.imread(f"/home/aolivepe/HW5/my_HW5_images_2/2.jpg")
    img3 = cv2.imread(f"/home/aolivepe/HW5/my_HW5_images_2/3.jpg")
    img4 = cv2.imread(f"/home/aolivepe/HW5/my_HW5_images_2/4.jpg")
    img5 = cv2.imread(f"/home/aolivepe/HW5/my_HW5_images_2/5.jpg")
    images = [img1, img2, img3, img4, img5]
        
    output_folder = "./mine_2/" # instructions or mine
    num_keypoints = 250
    refined_homographies = []
    homographies = []
    fig, axes = plt.subplots(1, 4, figsize=(8*4, 8))

    # Iterate over the given images
    for k in range(len(images)-1):
        print(f"IMAGES {k+1} AND {k+2}")
        image1 = images[k]
        image2 = images[k+1]

        # Use sift method to find keypoints and matches
        sift_detector = cv2.SIFT_create()
        kp1, vector1 = sift_detector.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
        kp2, vector2 = sift_detector.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
        raw_matches = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True).match(vector1, vector2)
        sorted_matches = []
        for match in raw_matches:
            sorted_matches.append(match)
            sorted_matches.sort(key=lambda m: m.distance)

        # Get homogeneous coordinates of the keypoints that have matched
        hc_kp1 = []
        hc_kp2 = []
        for i in range(len(sorted_matches)):
            hc_kp1.append((int(kp1[sorted_matches[i].queryIdx].pt[0]), int(kp1[sorted_matches[i].queryIdx].pt[1]), 1)) 
            hc_kp2.append((int(kp2[sorted_matches[i].trainIdx].pt[0]), int(kp2[sorted_matches[i].trainIdx].pt[1]), 1)) 
        hc_kp1 = np.array(hc_kp1)
        hc_kp2 = np.array(hc_kp2)

        # Generate image with the 2 images concatenated showing the matchings
        final_image = cv2.drawMatches(image1, kp1, image2, kp2, sorted_matches[:num_keypoints], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{output_folder}correspondences_{k}.jpg", final_image)
        
        # Use ransac algorithm to determine which matchings correspond to inliers and which ones correspond to outliers
        final_inliers_pos = ransac(hc_kp1[:num_keypoints], hc_kp2[:num_keypoints])

        # Get the inliers and outliers
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        sorted_matches = np.array(sorted_matches)
        inliers_matches = sorted_matches[:num_keypoints][final_inliers_pos]
        outliers_matches = np.delete(sorted_matches[:num_keypoints], final_inliers_pos)

        # These vectors will contain the tuples of the coordinates of the inliers and the outliers
        hc_inliers_kp1 = []
        hc_inliers_kp2 = []
        hc_outliers_kp1 = []
        hc_outliers_kp2 = []

        for match in inliers_matches:
            # Get the index of the matched inliers
            img1_idx = match.queryIdx  # Index in keypoints1
            img2_idx = match.trainIdx  # Index in keypoints2

            # Get the coordinates of the matched inliers
            (x1, y1) = kp1[img1_idx].pt  # Coordinates in image 1
            (x2, y2) = kp2[img2_idx].pt  # Coordinates in image 2

            # Add tuple of coordinates in the vector
            hc_inliers_kp1.append((int(x1), int(y1), 1))
            hc_inliers_kp2.append((int(x2), int(y2), 1))
            
        hc_inliers_kp1 = np.array(hc_inliers_kp1)
        hc_inliers_kp2 = np.array(hc_inliers_kp2)

        for match in outliers_matches:
            # Get the index of the matched outliers
            img1_idx = match.queryIdx  # Index in keypoints1
            img2_idx = match.trainIdx  # Index in keypoints2

            # Get the coordinates of the matched outliers
            (x1, y1) = kp1[img1_idx].pt  # Coordinates in image 1
            (x2, y2) = kp2[img2_idx].pt  # Coordinates in image 2

            # Add tuple of coordinates in the vector
            hc_outliers_kp1.append((int(x1), int(y1), 1))
            hc_outliers_kp2.append((int(x2), int(y2), 1))
            
        hc_outliers_kp1 = np.array(hc_outliers_kp1)
        hc_outliers_kp2 = np.array(hc_outliers_kp2)

        # Generate image with the 2 images concatenated showing the matchings between inliers and outliers
        final_image = putPointsAndLines(image1, image2, hc_inliers_kp1, hc_inliers_kp2, hc_outliers_kp1, hc_outliers_kp2)
        cv2.imwrite(f"{output_folder}inliers_outliers_{k}.jpg", final_image)

        # Use inliers to get the final homography
        final_h = obtain_h(hc_inliers_kp1 , hc_inliers_kp2)
        homographies.append(final_h)
        final_h = final_h.reshape(9)

        # Refine the final homography using our implementation of the Levemberg Marquardt (LM) algorithm
        final_h, C_p_vector = levemberg_marquardt(final_h, hc_inliers_kp1, hc_inliers_kp2)
        
        # Save homography to build panoramic later
        final_h = final_h.reshape(3, 3)
        refined_homographies.append(final_h)
        
        # Plot geometric error across iterations of the LM algorithm
        axes[k].plot(C_p_vector)
        axes[k].set_xlabel("LM iteration", fontsize=20)
        axes[k].set_ylabel("Geometric Error", fontsize=20)
        axes[k].set_title(f"Image {k+1} -> Image {k+2}: \n Initial error: {C_p_vector[0]:.2f}, Final error: {C_p_vector[-1]:.2f}", fontsize=20)
        print(f"Image {k+1} -> Image {k+2}: \n Initial error: {C_p_vector[0]:.2f}, Final error: {C_p_vector[-1]:.2f}")
        
    # Create panoramics and save them
    plt.savefig(f"{output_folder}error.jpg")    
    panoramic_ransac = ImageStitch(images, homographies)
    panoramic_after_LM = ImageStitch(images, refined_homographies)
    cv2.imwrite(f"{output_folder}panoramic_ransac.jpg", panoramic_ransac)
    cv2.imwrite(f"{output_folder}panoramic_after_LM.jpg", panoramic_after_LM)