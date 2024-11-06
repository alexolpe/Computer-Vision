import cv2
import numpy as np
from BitVector import BitVector
import math
import os
from sklearn.preprocessing import StandardScaler

# Function to convert RGB image to HSI
def rgb_to_hsi(image):
    # Convert image to float for precision
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Intensity calculation
    I = (R + G + B) / 3
    
    # Saturation calculation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 * min_rgb / (R + G + B + 1e-6))

    # Hue calculation
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(numerator / denominator)
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # Normalize hue to [0,1]
    
    return np.dstack((H, S, I))

# Function to compute Local Binary Pattern (LBP)
def computeLbp(img, R=1, P=8):
    # Create feature vector full of zeros
    lbp_hist = [0]*(P+2)
    
    # Iterate over all the pixels of the image except of the ones at distance R of the border to avoid problems with LBP implementation
    for h in range(R, img.shape[0] - R):
        for w in range(R, img.shape[1] - R):
            pattern = []
            # Get the value of each of the p points arround the center
            for p in range(P):
                k, l = h + R*math.cos(2*math.pi*p/P), w - R*math.sin(2*math.pi*p/P)
                k_base, l_base = int(k), int(l)
                if abs(k - k_base) < 1e-8 and abs(l - l_base) < 1e-8:
                    img_val_at_p = img[k_base, l_base]
                else:
                    delta_k = k - k_base
                    delta_l = l - l_base

                    if delta_l < 1e-8:
                        img_val_at_p = (1 - delta_k) * img[k_base, l_base] + delta_k * img[k_base + 1, l_base]
                    elif delta_k < 1e-8:
                        img_val_at_p = (1 - delta_l) * img[k_base, l_base] + delta_l * img[k_base, l_base + 1]
                    else:
                        img_val_at_p = (
                            delta_k * (1 - delta_l) * img[k_base + 1, l_base] +
                            delta_k * delta_l * img[k_base + 1, l_base + 1] +
                            (1 - delta_k) * (1 - delta_l) * img[k_base, l_base] +
                            (1 - delta_k) * delta_l * img[k_base, l_base + 1]
                        )
                
                # Threshold value at p compared to the value of the pixel in the center
                if img_val_at_p >= img[h, w]:
                    pattern.append(1)
                else:
                    pattern.append(0)
            
            # Similar to Prof. Kak's tutorial
            bv = BitVector(bitlist=pattern)
            int_circular_shift = [int(bv << i) for i in range(P)]
            minbv = BitVector(intVal=min(int_circular_shift), size=P)
            bvruns = minbv.runs()
            
            if len(bvruns) > 2:
                lbp_hist[P + 1] += 1
            elif len(bvruns) == 1 and bvruns[0][0] == '1':
                lbp_hist[P] += 1
            elif len(bvruns) == 1 and bvruns[0][0] == '0':
                lbp_hist[0] += 1
            else:
                lbp_hist[len(bvruns[1])] += 1
                
    return lbp_hist

def get_data(dir):
    feat = []
    lab = []
    
    # Iterate over all images
    for filename in sorted(os.listdir(dir)):
        image_path = os.path.join(dir, filename)
        
        #Read image, resize it, get hue 
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skip {image_path}")
            continue
        img = cv2.resize(img, (64, 64))
        hsi_image = rgb_to_hsi(img)
        hue_channel = (hsi_image[:, :, 0] * 255).astype(np.uint8)
 
        #Get LBP feature vector and append it to the features list 
        feat.append(computeLbp(hue_channel, R=1, P=8))
        
        # Append the integer corresponding to the label in the labels list
        classes = ["cloudy", "rain", "shine", "sunrise"]
        if 'cloudy' in filename:
            lab.append(classes.index("cloudy"))
        elif 'rain' in filename:
            lab.append(classes.index("rain"))
        elif 'shine' in filename:
            lab.append(classes.index("shine"))
        elif 'sunrise' in filename:
            lab.append(classes.index("sunrise"))
        else:
            continue
    
    feat = np.array(feat)
    lab = np.array(lab)
    
    # (feat - mean) / std_dev
    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)
    return feat, lab

train_dir = '/data/aolivepe/HW7-Auxilliary/data/training'
test_dir = '/data/aolivepe/HW7-Auxilliary/data/testing'  
    
# Get features and labels from the training and testing dataset and save them
train_feat, train_lab = get_data(train_dir)
np.savez_compressed(f'./results/lbp_train.npz', features=train_feat, labels=train_lab)

test_feat, test_lab = get_data(test_dir)
np.savez_compressed(f'./results/lbp_test.npz', features=test_feat, labels=test_lab)