import numpy as np
import skimage, skimage.color
import matplotlib.pyplot as plt

def get_row_disp(left_img, right_img, row_idx, patch_size, max_disparity):
    half_patch = patch_size // 2
    disparities = []
    disparity_range = np.arange(max_disparity)
    
    # Loop through each column within valid range
    for col in range(left_img.shape[1] - 2 * half_patch):
        # Define valid disparity range for the current column
        shifted_cols = col - disparity_range
        valid_shifts = (shifted_cols >= 0) & (shifted_cols < left_img.shape[1] - 2 * half_patch)
        valid_disparities = disparity_range[valid_shifts]
        
        # Extract window from the left image
        left_patch = left_img[row_idx:row_idx + patch_size, col:col + patch_size]
        binarized_left = left_patch > left_patch[half_patch, half_patch]
        
        # Evaluate disparity candidates
        cost_list = []
        for disparity in valid_disparities:
            right_patch = right_img[row_idx:row_idx + patch_size, col - disparity:col - disparity + patch_size]
            binarized_right = right_patch > right_patch[half_patch, half_patch]
            # Compute cost as binary mismatch count
            mismatch_count = np.sum(binarized_left != binarized_right)
            cost_list.append((disparity, mismatch_count))
        
        # Choose the disparity with the minimum cost
        optimal_disparity, _ = min(cost_list, key=lambda pair: pair[1])
        disparities.append(optimal_disparity)
    return np.array(disparities)

def main(img_l, img_r, window, max_dispar):
    window_2 = window // 2
    padded_im_left = np.pad(img_l, [(window_2, window_2)] * 2, mode='reflect')
    padded_im_right = np.pad(img_r, [(window_2, window_2)] * 2, mode='reflect')
    return np.asarray([get_row_disp(padded_im_left, padded_im_right, i, window, max_dispar) for i in range(img_l.shape[0])])

dm_gt = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp2.png")
print(dm_gt.shape)
dm_gt = (dm_gt.astype(np.float32) / 4.0).astype(np.uint8)

disparity_im_1 = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/im2.png")
disparity_im_2 = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/im6.png")

windows = [11, 15, 21, 31]
for window in windows:
    disp = main(skimage.color.rgb2gray(disparity_im_1),skimage.color.rgb2gray(disparity_im_2),window,52)
    plt.imshow(disp, cmap='gray')
    plt.axis('off')  # Remove axes for better appearance
    plt.savefig(f'/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disparity_{window}.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    mask_no_black = dm_gt > 0
    diff = np.abs(dm_gt - disp)
    delta = 2
    error_mask = (diff < delta) & mask_no_black
    accuracy = np.count_nonzero(diff[mask_no_black] < delta) / np.count_nonzero(mask_no_black)
    print(f"Accuracy for window {window}", accuracy)
    plt.imshow(error_mask, cmap='gray')
    plt.axis('off')  # Remove axes for better appearance
    plt.savefig(f'/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/error_{window}.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

