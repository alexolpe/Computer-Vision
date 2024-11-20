from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import numpy as np
import sympy as sp
import skimage, skimage.io, skimage.filters, skimage.transform, skimage.color, skimage.feature, skimage.measure, skimage.morphology
# from skimage.feature import plot_matched_features
import cv2
import matplotlib.pyplot as plt


class CensusTransform:
    def __init__(self,
        max_disparity=10,
        window_size=5
    ):
        self.max_disparity=max_disparity
        self.window_size = window_size
        assert self.window_size % 2 == 1
        self.window_size_2 = self.window_size // 2
    
    def pad_img(self, img):
        assert img.ndim == 2
        return np.pad(img, [(self.window_size_2, self.window_size_2)] * 2, mode='reflect')

    @staticmethod
    def transform_window(win):
        h = win.shape[0] // 2
        return win > win[h, h]

    def estimate_row_disparity(self, img_l_pad, img_r_pad, row_ind):
        col_size = img_l_pad.shape[1] - 2 * self.window_size_2
        disp_candidate = np.arange(self.max_disparity)
        ret = []
        
        for j in range(col_size):
            # compute valid disparity values
            right_col_loc = j - disp_candidate
            good_right_col_loc = np.bitwise_and(right_col_loc >= 0, right_col_loc < col_size)
            good_disp_candidate = disp_candidate[good_right_col_loc]
            win_l = img_l_pad[row_ind : row_ind + self.window_size, j : j + self.window_size]
            assert win_l.size == self.window_size ** 2
            assert win_l.shape[0] == win_l.shape[1]
            win_l_bin = self.transform_window(win_l)
            disp_ret = []
            
            for disp in good_disp_candidate:
                win_r = img_r_pad[row_ind : row_ind + self.window_size, j - disp : j - disp + self.window_size]
                assert win_l.shape == win_r.shape
                win_r_bin = self.transform_window(win_r)
                disp_ret.append((disp, np.count_nonzero(win_l_bin != win_r_bin)))
            
            best_disp, _ = min(disp_ret, key=lambda x: x[1])
            ret.append(best_disp)
            
        return np.asarray(ret)

    def run(self, img_l, img_r):
        assert img_l.shape == img_r.shape
        assert img_l.ndim == 2
        img_l_pad = self.pad_img(img_l)
        img_r_pad = self.pad_img(img_r)
        all_disp = [self.estimate_row_disparity(img_l_pad, img_r_pad, i) for i in range(img_l.shape[0])]
        return np.asarray(all_disp)

dm_ground_truth = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp2.png")
assert dm_ground_truth.dtype == np.uint8
dm_ground_truth = dm_ground_truth.astype(np.float32)
dm_ground_truth /= 4.0
dm_ground_truth = dm_ground_truth.astype(np.uint8)
print(dm_ground_truth.max())

dm_img_l = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/im2.png")
dm_img_r = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/im6.png")
ct = CensusTransform(window_size=7, max_disparity=52)
disp = ct.run(
skimage.color.rgb2gray(dm_img_l),
skimage.color.rgb2gray(dm_img_r)
)
non_black_mask = dm_ground_truth > 0
diff = np.abs(dm_ground_truth - disp)
acc = np.count_nonzero(diff[non_black_mask] < 2) / np.count_nonzero(non_black_mask)
print(acc)
plt.imshow(disp, cmap='jet')
plt.colorbar()
plt.axis('off')  # Remove axes for better appearance
plt.savefig('/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp_win7.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
# error mask
error_mask = np.zeros_like(non_black_mask)
error_mask[diff < 2] = True
error_mask[np.bitwise_not(non_black_mask)] = False
plt.imshow(error_mask, cmap='gray')
plt.axis('off')  # Remove axes for better appearance
plt.savefig('/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp_error_win7.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

dm_img_l = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/im2.png")
dm_img_r = skimage.io.imread("/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/im6.png")
ct = CensusTransform(window_size=9, max_disparity=52)
disp = ct.run(
skimage.color.rgb2gray(dm_img_l),
skimage.color.rgb2gray(dm_img_r)
)
non_black_mask = dm_ground_truth > 0
diff = np.abs(dm_ground_truth - disp)
acc = np.count_nonzero(diff[non_black_mask] < 2) / np.count_nonzero(non_black_mask)
print(acc)
plt.imshow(disp, cmap='jet')
plt.colorbar()
plt.axis('off')  # Remove axes for better appearance
plt.savefig('/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp_win9.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

# error mask
error_mask = np.zeros_like(non_black_mask)
error_mask[diff < 2] = True
error_mask[np.bitwise_not(non_black_mask)] = False
plt.imshow(error_mask, cmap='gray')
plt.axis('off')  # Remove axes for better appearance
plt.savefig('/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp_error_win9.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

plt.imshow(dm_ground_truth, cmap='jet')
plt.colorbar()
plt.axis('off')  # Remove axes for better appearance
plt.savefig('/home/aolivepe/Computer-Vision/HW9/Task3Images/Task3Images/disp_gt.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

