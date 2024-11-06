import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from vgg_and_resnet import VGG19, CustomResNet

def gram_matrix(feat_map):
    # Compute gram matrix given the feature map
    feats = feat_map.reshape(feat_map.shape[0], feat_map.shape[1] * feat_map.shape[2])
    G = np.dot(feats, feats.T)
    return np.log1p(G)    

test_dir = '/data/aolivepe/HW7-Auxilliary/data/testing'
vgg_model = VGG19()
vgg_model.load_weights('/data/aolivepe/HW7-Auxilliary/vgg_normalized.pth')
resnet_model = CustomResNet(encoder='resnet50')

fig, axs = plt.subplots(3, 4, figsize=(20, 15))
classes = ['cloudy', 'rain', 'shine', 'sunrise']

# Iterate over the 4 classes and randomly select one image for each class to plot the Gram matrix with each of the approaches used
for i, class_l in enumerate(classes):
    img_filenames = [im for im in os.listdir(test_dir) if im.lower().startswith(class_l)]
    if not img_filenames:
        continue   
    img = cv2.imread(os.path.join(test_dir, random.choice(img_filenames)))
    img = cv2.resize(img, (256, 256))

    vgg_feat = vgg_model(img)
    G = gram_matrix(vgg_feat)
    cax = axs[0, i].imshow(G, cmap='viridis')
    fig.colorbar(cax, ax=axs[0, i])
    axs[0, i].set_title(f"Method: Vgg, Class: {class_l.capitalize()}")

    _, resnet_fine_feat = resnet_model(img)
    G = gram_matrix(resnet_fine_feat)
    cax = axs[1, i].imshow(G, cmap='viridis', vmin=0.75, vmax=4)
    fig.colorbar(cax, ax=axs[1, i])
    axs[1, i].set_title(f"Method: Resnet_fine, Class: {class_l.capitalize()}")

    resnet_coarse_feat, _ = resnet_model(img)
    G = gram_matrix(resnet_coarse_feat)
    cax = axs[2, i].imshow(G, cmap='viridis', vmin=0.0, vmax=1)
    fig.colorbar(cax, ax=axs[2, i])
    axs[2, i].set_title(f"Method: Resnet_coarse, Class: {class_l.capitalize()}")

plt.tight_layout()
# Save the result
plt.savefig("./results/gram_matrices.jpg")