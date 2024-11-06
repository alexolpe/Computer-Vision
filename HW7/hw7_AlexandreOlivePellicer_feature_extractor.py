import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from vgg_and_resnet import VGG19, CustomResNet

def gram_matrix(feat_map):
    # Compute gram matrix given the feature map
    feats = feat_map.reshape(feat_map.shape[0], feat_map.shape[1] * feat_map.shape[2])
    G = np.dot(feats, feats.T)
    gram = G / (feat_map.shape[0] * feat_map.shape[1] * feat_map.shape[2])
    
    # Downsample to 32x32 as mentioned in instructions
    down_gram = cv2.resize(gram, (32, 32), interpolation=cv2.INTER_LINEAR)
    return down_gram[np.triu_indices(32)]

def get_data(dir, model, method):
    feat = []
    lab = []
    
    # Iterate over all images
    for filename in sorted(os.listdir(dir)):
        image_path = os.path.join(dir, filename)
        
        #Read image, resize it, conevrt it to RGB and scale between 0 and 1        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skip {image_path}")
            continue
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        
        # Get feature vector from the Gram Matrix according to the method used and append it to the features list 
        if method == 'vgg':
            vgg_feat = model(img)
            feat.append(gram_matrix(vgg_feat))
        elif method == 'resnet_coarse':
            resnet_coarse_feat, _ = model(img)
            feat.append(gram_matrix(resnet_coarse_feat))
        elif method == 'resnet_fine':
            _, resnet_fine_feat = model(img)
            feat.append(gram_matrix(resnet_fine_feat))
        
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

# Load pretrained VGG and ResNet
vgg_model = VGG19()
vgg_model.load_weights('/data/aolivepe/HW7-Auxilliary/vgg_normalized.pth')
resnet_model = CustomResNet(encoder='resnet50')

# Get features and labels from the training and testing dataset and save them. Do this for each feature extractor method used
method = "vgg"
train_feat, train_lab = get_data(train_dir, vgg_model, method)
np.savez_compressed(f'./results/{method}_train.npz', features=train_feat, labels=train_lab)
test_feat, test_lab = get_data(test_dir, vgg_model, method)
np.savez_compressed(f'./results/{method}_test.npz', features=test_feat, labels=test_lab)
print(f"{method} train and test completed")

method = "resnet_coarse"
train_feat, train_lab = get_data(train_dir, resnet_model, method)
np.savez_compressed(f'./results/{method}_train.npz', features=train_feat, labels=train_lab)
test_feat, test_lab = get_data(test_dir, resnet_model, method)
np.savez_compressed(f'./results/{method}_test.npz', features=test_feat, labels=test_lab)
print(f"{method} train and test completed")

method = "resnet_fine"
train_feat, train_lab = get_data(train_dir, resnet_model, method)
np.savez_compressed(f'./results/{method}_train.npz', features=train_feat, labels=train_lab)
test_feat, test_lab = get_data(test_dir, resnet_model, method)
np.savez_compressed(f'./results/{method}_test.npz', features=test_feat, labels=test_lab)
print(f"{method} train and test completed")


