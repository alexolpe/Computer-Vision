import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class WeakClassifier:
    def __init__(self, features, labels, weights):
        self.features = features
        self.labels = labels
        self.weights = weights
        self.best_feature_idx = None
        self.best_threshold = None
        self.best_polarity = None
        self.best_error = float("inf")
        self.best_trust_factor = None

    def findBestClassifier(self):
        for feature_idx in range(self.features.shape[1]):
            sorted_indices = np.argsort(self.features[:, feature_idx])
            sorted_features = self.features[sorted_indices, feature_idx]
            sorted_labels = self.labels[sorted_indices]
            sorted_weights = self.weights[sorted_indices]
    
            pos_weights = np.cumsum(sorted_weights * (sorted_labels == 1))
            neg_weights = np.cumsum(sorted_weights * (sorted_labels == 0))
    
            total_pos_weight = np.sum(sorted_weights[sorted_labels == 1])
            total_neg_weight = np.sum(sorted_weights[sorted_labels == 0])
    
            error_polarity_1 = pos_weights + (total_neg_weight - neg_weights)
            error_polarity_2 = neg_weights + (total_pos_weight - pos_weights)
    
            errors = np.vstack((error_polarity_1, error_polarity_2)).min(axis=0)
            min_error_idx = errors.argmin()
    
            if errors[min_error_idx] < self.best_error:
                self.best_error = errors[min_error_idx]
                self.best_feature_idx = feature_idx
                self.best_threshold = sorted_features[min_error_idx]
                self.best_polarity = 1 if error_polarity_1[min_error_idx] < error_polarity_2[min_error_idx] else -1
    
        self.best_trust_factor = 0.5 * np.log((1 - self.best_error) / (self.best_error + 1e-10))


    def classify(self, data):
        feature_values = data[:, self.best_feature_idx]
        if self.best_polarity == 1:
            return (feature_values >= self.best_threshold).astype(int)
        else:
            return (feature_values < self.best_threshold).astype(int)


class Adaboost:
    def __init__(self, num_cascades, iterations_per_cascade, tolerance=1e-6):
        self.num_cascades = num_cascades
        self.iterations_per_cascade = iterations_per_cascade
        self.tolerance = tolerance
        self.classifiers_per_cascade = {}
        self.fprs = []
        self.fnrs = []

    def extractFeatures(self, directory):
        features = []
        for file_name in os.listdir(directory):
            if file_name.endswith(".png"):
                img_path = os.path.join(directory, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features.append(self.calculateFeatures(img))
        return np.array(features)

    def calculateFeatures(self, img):
        integral_img = cv2.integral(img)
        features = []
        #I'm using Haar features
        for y in range(0, img.shape[0] - 4, 4):  
            for x in range(0, img.shape[1] - 4, 4):  
                for height in range(4, img.shape[0] - y, 4):
                    for width in range(4, img.shape[1] - x, 4):
                        # Horizontal edge feature
                        left_sum = integral_img[y, x] + integral_img[y + height, x + width // 2] - integral_img[y, x + width // 2] - integral_img[y + height, x]
                        right_sum = integral_img[y, x + width // 2] + integral_img[y + height, x + width] - integral_img[y, x + width] - integral_img[y + height, x + width // 2]
                        horizontal_edge_feature = right_sum - left_sum
                        features.append(horizontal_edge_feature)

                        # Vertical edge feature
                        top_sum = integral_img[y, x] + integral_img[y + height // 2, x + width] - integral_img[y + height // 2, x] - integral_img[y, x + width]
                        bottom_sum = integral_img[y + height // 2, x] + integral_img[y + height, x + width] - integral_img[y + height, x] - integral_img[y + height // 2, x + width]
                        vertical_edge_feature = bottom_sum - top_sum
                        features.append(vertical_edge_feature)

                        # Four-rectangle feature
                        top_left = integral_img[y, x] + integral_img[y + height // 2, x + width // 2] - integral_img[y, x + width // 2] - integral_img[y + height // 2, x]
                        top_right = integral_img[y, x + width // 2] + integral_img[y + height // 2, x + width] - integral_img[y, x + width] - integral_img[y + height // 2, x + width // 2]
                        bottom_left = integral_img[y + height // 2, x] + integral_img[y + height, x + width // 2] - integral_img[y + height // 2, x + width // 2] - integral_img[y + height, x]
                        bottom_right = integral_img[y + height // 2, x + width // 2] + integral_img[y + height, x + width] - integral_img[y + height // 2, x + width] - integral_img[y + height, x + width // 2]
                        four_rectangle_feature = (top_left + bottom_right) - (top_right + bottom_left)
                        features.append(four_rectangle_feature)

        return np.array(features)

    def createLabels(self, pos_features, neg_features):
        features = np.vstack((pos_features, neg_features))
        labels = np.hstack((np.ones(pos_features.shape[0]), np.zeros(neg_features.shape[0])))
        return features, labels

    def trainCascade(self, train_features, train_labels):
        weights = np.hstack((np.ones(np.sum(train_labels == 1)) / np.sum(train_labels == 1),
                             np.ones(np.sum(train_labels == 0)) / np.sum(train_labels == 0)))
        cumulative_fpr = 1.0
        cumulative_fnr = 1.0
    
        for cascade_level in range(self.num_cascades):
            print(f"Training cascade level {cascade_level + 1}")
            best_classifier = None
            best_fpr, best_fnr = None, None
    
            for _ in range(self.iterations_per_cascade):
                weak_clf = WeakClassifier(train_features, train_labels, weights)
                weak_clf.findBestClassifier()
                predictions = weak_clf.classify(train_features)
                error = (weights * np.abs(predictions - train_labels)).sum()
    
                if error > 0.5:
                    continue
    
                weights *= np.exp(-weak_clf.best_trust_factor * train_labels * (2 * predictions - 1))
                weights /= weights.sum()
    
                false_positive_rate = predictions[train_labels == 0].mean()
                false_negative_rate = 1 - predictions[train_labels == 1].mean()
    
                if not best_classifier or false_positive_rate < best_fpr:
                    best_classifier = weak_clf
                    best_fpr = false_positive_rate
                    best_fnr = false_negative_rate
    
            self.classifiers_per_cascade[cascade_level] = best_classifier
            cumulative_fpr *= best_fpr
            cumulative_fnr *= best_fnr
    
            self.fprs.append(cumulative_fpr)
            self.fnrs.append(cumulative_fnr)
    
            print(f"Cascade Level: {cascade_level + 1}, Cumulative FPR: {cumulative_fpr:.4f}, Cumulative FNR: {cumulative_fnr:.4f}")
    
            if cumulative_fpr < self.tolerance:
                print(f"Tolerance reached at cascade level {cascade_level + 1}")
                break

    def plotPerformance(self, save_dir, title):
        plt.figure()
        plt.plot(self.fprs, label="FPR", marker="o")
        plt.plot(self.fnrs, label="FNR", marker="x")
        plt.xlabel("Cascade Level")
        plt.ylabel("Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "adaboost_performance.png"))
        plt.close()


def plotFPRvsStages(fprs, save_dir, title="FPR vs Stages"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fprs) + 1), fprs, marker='o', label="FPR")
    plt.xlabel("Cascade Stages")
    plt.ylabel("False Positive Rate (FPR)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fpr_vs_stages.png"))
    plt.show()

def main():
    train_dir = './CarDetection/train'
    test_dir = './CarDetection/test'
    result_dir = './results/Adaboost'

    os.makedirs(result_dir, exist_ok=True)

    adaboost = Adaboost(num_cascades=10, iterations_per_cascade=10)

    # Extract features for training
    print("Extracting features from training set...")
    pos_train_features = adaboost.extractFeatures(os.path.join(train_dir, 'positive'))
    neg_train_features = adaboost.extractFeatures(os.path.join(train_dir, 'negative'))
    train_features, train_labels = adaboost.createLabels(pos_train_features, neg_train_features)

    # Train AdaBoost Cascade
    adaboost.trainCascade(train_features, train_labels)
    adaboost.plotPerformance(result_dir, "Adaboost Cascade Performance")
    
    plotFPRvsStages(adaboost.fprs, result_dir, "FPR vs Stages During Training")
    # Extract features for testing
    print("Extracting features from test set...")
    pos_test_features = adaboost.extractFeatures(os.path.join(test_dir, 'positive'))
    neg_test_features = adaboost.extractFeatures(os.path.join(test_dir, 'negative'))
    test_features, test_labels = adaboost.createLabels(pos_test_features, neg_test_features)

    # Evaluate the cascade on test data
    print("Evaluating on test set...")
    test_predictions = []
    for cascade_level, classifier in adaboost.classifiers_per_cascade.items():
        predictions = classifier.classify(test_features)
        test_predictions.append(predictions)

    final_predictions = np.array(test_predictions).mean(axis=0)
    accuracy = (final_predictions == test_labels).mean() * 100
    print(f"Final Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
