import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
from PIL import Image

def get_confusion_matrix(train_feat, train_lab, test_feat, test_lab, path, method):
    # train svm classifier with training data
    svm_class = SVC(kernel='linear')
    svm_class.fit(train_feat, train_lab)
    
    # get prediction with testing data
    pred = svm_class.predict(test_feat)
    
    # create confusion matrix
    cm = confusion_matrix(test_lab, pred)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(path)
    
    class_names = ["cloudy", "rain", "shine", "sunrise"]
    test_dir = '/data/aolivepe/HW7-Auxilliary/data/testing'
    image_paths = [os.path.join(test_dir, img) for img in sorted(os.listdir(test_dir))]
    # Iterate over each class and find correct and incorrect classifications
    for cls in range(4):
        class_idx = np.where(test_lab == cls)[0]
        correct_idx = class_idx[test_lab[class_idx] == pred[class_idx]]
        incorrect_idx = class_idx[test_lab[class_idx] != pred[class_idx]]
        
        # Save one example of correct and incorrect classification
        if len(correct_idx) > 0:
            # Read image from path
            correct_image = Image.open(image_paths[correct_idx[0]])
            save_annotated_image(correct_image, test_lab[correct_idx[0]], pred[correct_idx[0]], True, class_names, correct_idx[0], method)
        
        if len(incorrect_idx) > 0:
            # Read image from path
            incorrect_image = Image.open(image_paths[incorrect_idx[0]])
            save_annotated_image(incorrect_image, test_lab[incorrect_idx[0]], pred[incorrect_idx[0]], False, class_names, incorrect_idx[0], method)
        
# Annotate and save images
def save_annotated_image(img, gt, pred, correct, class_name, idx, method):
    img = img.copy()
    text = f"GT: {class_name[gt]}, Pred: {class_name[pred]}, {'Correct' if correct else 'Incorrect'}"
    filename = f"/home/aolivepe/HW7/results/{method}_{text}.jpg"
    img.save(filename, format="JPEG")


# Load the feature vectors and labels of the training and testing dataset for each method and compute the confusion matrix
test = np.load('/home/aolivepe/HW7/results/lbp_test.npz')
test_feat = test['features']
test_lab = test['labels']

train = np.load('/home/aolivepe/HW7/results/lbp_train.npz')
train_feat = train['features']
train_lab = train['labels']

get_confusion_matrix(train_feat, train_lab, test_feat, test_lab, "/home/aolivepe/HW7/results/lbp_confusion_matrix.jpg", "lbp")

test = np.load('/home/aolivepe/HW7/results/vgg_test.npz')
test_feat = test['features']
test_lab = test['labels']

train = np.load('/home/aolivepe/HW7/results/vgg_train.npz')
train_feat = train['features']
train_lab = train['labels']

get_confusion_matrix(train_feat, train_lab, test_feat, test_lab, "/home/aolivepe/HW7/results/vgg_confusion_matrix.jpg", "vgg")

test = np.load('/home/aolivepe/HW7/results/resnet_fine_test.npz')
test_feat = test['features']
test_lab = test['labels']

train = np.load('/home/aolivepe/HW7/results/resnet_fine_train.npz')
train_feat = train['features']
train_lab = train['labels']

get_confusion_matrix(train_feat, train_lab, test_feat, test_lab, "/home/aolivepe/HW7/results/resnet_fine_confusion_matrix.jpg", "resnet_fine")

test = np.load('/home/aolivepe/HW7/results/resnet_coarse_test.npz')
test_feat = test['features']
test_lab = test['labels']

train = np.load('/home/aolivepe/HW7/results/resnet_coarse_train.npz')
train_feat = train['features']
train_lab = train['labels']

get_confusion_matrix(train_feat, train_lab, test_feat, test_lab, "/home/aolivepe/HW7/results/resnet_coarse_confusion_matrix.jpg", "resnet_coarse")

