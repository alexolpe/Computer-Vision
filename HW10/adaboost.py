import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_best_class(features, labels, wg):
    num_features = features.shape[1]
    best_params = {"feature_idx": None,"threshold": None,"polarity": None,"error": float("inf"),"trust_factor": None}

    for feature_idx in range(num_features):
        # Sort feature values and associated labels/wg
        sorted_data = sorted(
            zip(features[:, feature_idx], labels, wg), key=lambda x: x[0]
        )
        sorted_features, sorted_labels, sorted_wg = zip(*sorted_data)

        # Precompute total positive and negative wg
        total_pos_weight = sum(w for l, w in zip(sorted_labels, sorted_wg) if l == 1)
        total_neg_weight = sum(w for l, w in zip(sorted_labels, sorted_wg) if l == 0)

        pos_cumsum = 0
        neg_cumsum = 0

        # Traverse through thresholds and calculate errors dynamically
        for i in range(len(sorted_features)):
            if sorted_labels[i] == 1:
                pos_cumsum += sorted_wg[i]
            else:
                neg_cumsum += sorted_wg[i]

            # Calculate errors for both polarities
            error_polarity_1 = pos_cumsum + (total_neg_weight - neg_cumsum)
            error_polarity_2 = neg_cumsum + (total_pos_weight - pos_cumsum)

            # Select the polarity with the smaller error
            if error_polarity_1 < error_polarity_2:
                error = error_polarity_1
                polarity = 1
            else:
                error = error_polarity_2
                polarity = 0

            # Update best parameters if current error is smaller
            if error < best_params["error"]:
                best_params.update({"feature_idx": feature_idx, "threshold": sorted_features[i],"polarity": polarity,"error": error})
    # Compute the trust factor for the best classifier
    best_params["trust_factor"] = 0.5 * np.log((1 - best_params["error"]) / (best_params["error"] + 1e-10))
    return best_params

# Preapare data
def getData(dir):
    # Positive data
    features = []
    for file_name in os.listdir(os.path.join(dir, 'positive')):
        img_path = os.path.join(dir, 'positive', file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        features.append(getFeatures(img))
    pos_features = np.array(features)
    
    # Negative data
    features = []
    for file_name in os.listdir(os.path.join(dir, 'negative')):
        img_path = os.path.join(dir, 'negative', file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        features.append(getFeatures(img))
    neg_features = np.array(features)
    
    # All data and labels
    features = np.vstack((pos_features, neg_features))
    labels = np.hstack((np.ones(pos_features.shape[0]), np.zeros(neg_features.shape[0])))
    return features, labels

# Get Haar features
def getFeatures(img):
    def compute_integral_sum(intimg, x1, y1, x2, y2):
        return (intimg[y2, x2] - intimg[y1, x2] 
                - intimg[y2, x1] + intimg[y1, x1])
    
    intimg = cv2.integral(img)
    features = []

    # Generate grid points for top-left corner of patches
    y_coords, x_coords = np.meshgrid(
        np.arange(0, img.shape[0] - 4, 4),
        np.arange(0, img.shape[1] - 4, 4),
        indexing="ij"
    )

    start_points = np.column_stack((y_coords.ravel(), x_coords.ravel()))

    # Loop over each starting point
    for y, x in start_points:
        # Loop over height and width increments
        for height in range(4, img.shape[0] - y, 4):
            for width in range(4, img.shape[1] - x, 4):
                # Horizontal and vertical features
                features.append(compute_integral_sum(intimg, x + width // 2, y, x + width, y + height) - compute_integral_sum(intimg, x, y, x + width // 2, y + height))
                features.append(compute_integral_sum(intimg, x, y + height // 2, x + width, y + height) - compute_integral_sum(intimg, x, y, x + width, y + height // 2))
    return np.array(features)

def cascade_classifiers(features, labels, num_cascades = 10, iterat = 10):
    classifiers_per_cascade = {}
    fprs = []
    fnrs = []
    fpr_c = 1.0
    fnr_c = 1.0
    
    # Calculate initial weights for positive and negative samples
    positive_wg = np.ones(np.sum(labels == 1)) / np.sum(labels == 1)
    negative_wg = np.ones(np.sum(labels == 0)) / np.sum(labels == 0)
    wg = np.hstack((positive_wg, negative_wg))

    # Loop through cascade levels
    for level in range(num_cascades):
        # Find the best classifier for the current cascade level
        best_classifier, fpr_b, fnr_b = _findBestClassifierForCascade(features, labels, wg, iterat)

        # Store the best classifier for this cascade level
        classifiers_per_cascade[level] = best_classifier
        fpr_c *= fpr_b
        fnr_c *= fnr_b
        fprs.append(fpr_c)
        fnrs.append(fnr_c)

        print(fpr_c, fnr_c)
        # Stop training if tolerance is reached
        if fpr_c < 1e-6:
            break
    return fprs, fnrs, classifiers_per_cascade

def _findBestClassifierForCascade(features, labels, wg, iter_casc= 10):
    best_classifier = None
    fpr_b, fnr_b = float("inf"), float("inf")
    
    for _ in range(iter_casc):
        # Find the best weak classifier
        classif = get_best_class(features, labels, wg)

        # Classify using the best weak classifier
        pred = features[:, classif["feature_idx"]] >= classif["threshold"] if classif["polarity"] == 1 else features[:, classif["feature_idx"]] < classif["threshold"]

        # Update weights
        wg = wg * np.exp(-classif["trust_factor"] * labels * (2 * pred - 1))
        wg = wg / wg.sum()

        # Calculate false positive and false negative rates
        fpr = pred[labels == 0].mean()
        fnr = 1 - pred[labels == 1].mean()

        # Update the best classifier if FPR improves
        if fpr < fpr_b:
            best_classifier = classif
            fpr_b = fpr
            fnr_b = fnr

    return best_classifier, fpr_b, fnr_b

# Plot required plots
def plotFprFnr(fprs, fnrs, out_dir):
    plt.figure(figsize=(14, 6))
    stages = range(1, len(fprs) + 1)
    
    # Plot only FPR
    plt.subplot(1, 2, 1)
    plt.plot(stages, fprs, marker='s', color="red", label="FPR")
    plt.xlabel("Cascade stage")
    plt.title("Evolution of FPR through stages")
    plt.grid(True)
    plt.legend()
    
    # Plot FPR and FNR
    plt.subplot(1, 2, 2)
    plt.plot(stages, fprs, marker='s', color="red", label="FPR")
    plt.plot(stages, fnrs, marker="p", color="green", label="FNR")
    plt.xlabel("Cascade stage")
    plt.title("Evolution of FPR and FNR through stages")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(out_dir)

if __name__ == "__main__":
    num_cascades = 10
    iterat_casc = 5 #10
    fprs_t = []
    fnrs_t = []
    fpr_t = 1.0
    fnr_t = 1.0
    
    result_dir = '/home/aolivepe/Computer-Vision/HW10/results/Adaboost'

    # Prepare data and labels
    train_features, train_labels = getData('/home/aolivepe/Computer-Vision/HW10/CarDetection/train')
    test_features, test_labels = getData('/home/aolivepe/Computer-Vision/HW10/CarDetection/test')

    # Train cascade
    fprs, fnrs, classifiers_per_cascade = cascade_classifiers(train_features, train_labels, num_cascades, iterat_casc)

    # Test on testing dataset
    for cascade_level, classifier in classifiers_per_cascade.items():
        pred = test_features[:, classifier["feature_idx"]] >= classifier["threshold"] if classifier["polarity"] == 1 else test_features[:, classifier["feature_idx"]] < classifier["threshold"]
        fpr_t *= pred[test_labels == 0].mean()
        fnr_t *= 1 - pred[test_labels == 1].mean()
        fprs_t.append(fpr_t)
        fnrs_t.append(fnr_t)

    # Plot results for training and testing
    plotFprFnr(fprs, fnrs, result_dir+f"/fpr_fnr_train_{num_cascades}_{iterat_casc}.jpg")
    plotFprFnr(fprs_t, fnrs_t, result_dir+f"/fpr_fnr_test_{num_cascades}_{iterat_casc}.jpg")
