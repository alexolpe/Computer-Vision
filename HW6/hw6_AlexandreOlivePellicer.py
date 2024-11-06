import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

def otsu_thr(masked_image):
    # Compute histogram and bins
    hist, bin_edges = np.histogram(masked_image, bins=256, range=(0, 255), density=True) #it already returns the prob because den=true
    sig_prev = 0
    thr = 0
    
    # Loop through bin edges
    for b in range(256):
        # b goes from 0 to 255
        # k goes from 1 to 256
        k = b+1
        w0 = np.sum(hist[:b])
        w1 = np.sum(hist[b:])
                
        # Skip if either class weight is 0
        if w0 == 0 or w1 == 0:
            continue
        
        # Compute means
        mu0 = np.sum(np.arange(1, k, 1) * hist[:b]) / w0
        mu1 = np.sum(np.arange(k, 257, 1) * hist[b:]) / w1
        
        # Compute between-class variance
        sig = w0 * w1 * (mu1 - mu0)**2
        
        # Update thr if better variance found
        if sig >= sig_prev:
            sig_prev = sig
            thr = b
            
    return thr

def gray_otsu(image, it, inv):
    # Create mask of 1s and later we will substitute to 0 some values
    mask = np.ones(image.shape, dtype=bool)
    
    for i in range(it):
        # Get the threshold
        thr = otsu_thr(masked_image = image[mask])
        
        # Apply threshold to update mask. May need to inverse criteria depending on the colorcomposition of the image
        if inv:
            mask[image > thr] = 0
        else:
            mask[image < thr] = 0
    
    return mask

def rgb_otsu(image, it, inv, pic, folder = "rgb"): 
    # Get the binary mask for each channel (B, G, R)
    mask = np.zeros(image.shape, dtype=int)
    for i in range(image.shape[2]):
        image_channel = image[:, :, i]
        mask[:, :, i] = gray_otsu(image_channel, it[i], inv[i])
        cv2.imwrite(f"{folder}/mask_{pic}_{str(i)}.jpg", mask[:, :, i]*255)

    # Use and operator to combine the 3 masks
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    full_mask[(mask[:, :, 0] == 1) & (mask[:, :, 1] == 1) & (mask[:, :, 2] == 1)] = 1

    return full_mask

def normalize_to_255(features_map):
    # Function to normalize image between 0 and 255
    min_val = np.min(features_map)
    max_val = np.max(features_map)
    # Normalize to [0, 1]
    normalized = (features_map - min_val) / (max_val - min_val)
    # Scale to [0, 255]
    normalized_255 = normalized * 255
    return normalized_255

def texture_otsu(image, ns, it, inv, pic):
    texture_arr = []
    for i, n in enumerate(ns):
        # Add a padding of zeros to deal with the case when the filter is in the borders of the image
        padding = n//2
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), 'constant', constant_values=0)
        
        # Create a feature map computing the variance inside the window of size n x n
        features_map = np.zeros((image.shape[0], image.shape[1]))
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                var = np.var(padded_image[r:r+2*padding+1, c:c+2*padding+1])
                features_map[r, c] = var
        features_map = normalize_to_255(features_map)
        cv2.imwrite(f"textures/{pic}_features_{n}.jpg", features_map)
        texture_arr.append(features_map)
    
    # Stack the 3 feature maps and pass it to the rgb_otsu function to operate as it was done with the B, G, R channels of the original image
    texture_image = np.stack(texture_arr, axis=2)
    cv2.imwrite(f"textures/{pic}_texture_image.jpg", texture_image)
    return rgb_otsu(texture_image, it, inv, pic, folder = "textures")

def erosion(image):
    # 3x3 window as mask
    mask = np.ones((3, 3), dtype=np.uint8)
    mask_height, mask_width = mask.shape
    
    # Create eroded image
    eroded_image = np.zeros_like(image)
    
    # Add padding to the image to deal with borders
    padded_image = np.pad(image, ((mask_height//2, mask_height//2), (mask_width//2, mask_width//2)), mode='constant', constant_values=255)
    
    # Perform erosion
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # roi is the pixels in the mask
            roi = padded_image[i:i+mask_height, j:j+mask_width]
            # Apply the mask and take the minimum value
            eroded_image[i, j] = np.min(roi[mask == 1])
    
    return eroded_image

def dilation(image):
    # 3x3 window as mask
    mask = np.ones((3, 3), dtype=np.uint8)
    mask_height, mask_width = mask.shape
    
    # Create dilated image
    dilated_image = np.zeros_like(image)
    
    # Add padding to the image to deal with borders
    padded_image = np.pad(image, ((mask_height//2, mask_height//2), (mask_width//2, mask_width//2)), mode='constant', constant_values=0)
    
    # Perform dilation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # roi is the pixels in the mask
            roi = padded_image[i:i+mask_height, j:j+mask_width]
            # Apply the mask and take the maximum value
            dilated_image[i, j] = np.max(roi[mask == 1])
    
    return dilated_image

def extract_contours(binary_image):
    contour_mask = np.zeros_like(binary_image)
    rows, cols = binary_image.shape
    
    # Iterate through each pixel in the image excluding the borders
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # If the current pixel is part of the object (value 1)
            if binary_image[i, j] == 1:
                # Check the eight neighbors for at least one 0
                neighborhood = binary_image[i-1:i+2, j-1:j+2]
                if 0 in neighborhood:
                    contour_mask[i, j] = 1
    
    return contour_mask

## TASK 1.1 AND 1.3 -----------------------------------
pic = "dog"
image = cv2.imread(f"pics/{pic}_small.jpg")
full_mask=rgb_otsu(image, [1, 2, 2], [1,1,1], pic)
cv2.imwrite(f"rgb/{pic}.jpg", full_mask *255)

# Apply opening
eroded = erosion(full_mask)
dilated = dilation(eroded)
cv2.imwrite(f'contour/dilated_{pic}.jpg', dilated*255)
# Extract contours
contours = extract_contours(dilated)
cv2.imwrite(f'contour/contours_{pic}.jpg', contours * 255)

pic = "flower"
image = cv2.imread(f"pics/{pic}_small.jpg")
full_mask=rgb_otsu(image, [1, 2, 2], [0, 0, 0], pic)
cv2.imwrite(f"rgb/{pic}.jpg", full_mask *255)

eroded = erosion(full_mask)
dilated = dilation(eroded)
cv2.imwrite(f'contour/dilated_{pic}.jpg', dilated*255)
contours = extract_contours(dilated)
cv2.imwrite(f'contour/contours_{pic}.jpg', contours * 255)

pic = "car"
image = cv2.imread(f"pics/{pic}_small.jpg")
full_mask=rgb_otsu(image, [2, 1, 1], [0, 0, 0], pic)
cv2.imwrite(f"rgb/{pic}.jpg", full_mask *255)

eroded = erosion(full_mask)
dilated = dilation(eroded)
cv2.imwrite(f'contour/dilated_{pic}.jpg', dilated*255)
contours = extract_contours(dilated)
cv2.imwrite(f'contour/contours_{pic}.jpg', contours * 255)

pic = "squirrel"
image = cv2.imread(f"pics/{pic}_small.jpg")
full_mask=rgb_otsu(image, [1, 1, 1], [0, 0, 0], pic)
cv2.imwrite(f"rgb/{pic}.jpg", full_mask *255)

eroded = erosion(full_mask)
dilated = dilation(eroded)
cv2.imwrite(f'contour/dilated_{pic}.jpg', dilated*255)
contours = extract_contours(dilated)
cv2.imwrite(f'contour/contours_{pic}.jpg', contours * 255)

## TASK 1.2 -----------------------------------
pic = "dog"
image = cv2.imread(f"pics/{pic}_small.jpg", cv2.IMREAD_GRAYSCALE)
full_mask=texture_otsu(image, [5 ,7, 9], [3, 3, 3], [1,1,1], pic)
cv2.imwrite(f"textures/{pic}.jpg", full_mask *255)

pic = "flower"
image = cv2.imread(f"pics/{pic}_small.jpg", cv2.IMREAD_GRAYSCALE)
full_mask=texture_otsu(image, [13 ,15, 17], [1, 1, 1], [0, 0, 0], pic)
cv2.imwrite(f"textures/{pic}.jpg", full_mask *255)

pic = "car"
image = cv2.imread(f"pics/{pic}_small.jpg", cv2.IMREAD_GRAYSCALE)
full_mask=texture_otsu(image, [7 ,9, 11], [1, 1, 1], [0, 0, 0], pic)
cv2.imwrite(f"textures/{pic}.jpg", full_mask *255)

pic = "squirrel"
image = cv2.imread(f"pics/{pic}_small.jpg", cv2.IMREAD_GRAYSCALE)
full_mask=texture_otsu(image, [7, 9, 11], [1, 1, 1], [0, 0, 0], pic)
cv2.imwrite(f"textures/{pic}.jpg", full_mask *255)
