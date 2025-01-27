import numpy as np
import matplotlib.pyplot as plt
import cv2, os
import heapq
import pickle
import cv2
import numpy as np
import heapq
import seaborn as sns

# Prepare images and labels
def getData(path):
    images = []
    labels = []
    for num in range(30):
        for img in range(21):
            file_name = f"{(num+1):02}_{(img+1):02}.png"
            image_path = f"{path}/{file_name}"
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (32, 32))
            images.append(image.flatten())
            labels.append(num+1)
    return np.array(images).T, np.array(labels)

# Calculate covariance matrix
def getCovMatrix(data):
    C = data @ data.T
    C_compr = data.T @ data
    return C, C_compr

# Find large eigenvectors
def largeEig(data, C_compr, stg):
    eigenvalues, eigenvectors = np.linalg.eig(C_compr)
    transformed_vectors = np.real(np.dot(data, eigenvectors))
    transformed_vectors /= np.linalg.norm(transformed_vectors, axis=0)
    largest_eigenvalues = heapq.nlargest(stg, np.abs(eigenvalues))
    largest_indices = [np.where(np.abs(eigenvalues) == value)[0][0] for value in largest_eigenvalues]
    return transformed_vectors[:, largest_indices]

# Get real and predicted embeddings
def getCoeff(x_train, x_test, large_eig):
    real_c = x_train.T @ large_eig
    pred_c = x_test.T @ large_eig
    return real_c, pred_c

# Classify and calculate accuracy
def getAccuracy(true_labels, ground_truth_coeff, predicted_coeff, n_classes=30):
    # Calculate Euclidean distance
    def pairwise_distance(gt, pred):
        return np.sqrt(((gt[None, :, :] - pred[:, None, :]) ** 2).sum(axis=2))

    # Classify based on nearest neighbours
    def classify_based_on_distance(distance_matrix, labels):
        class_count = np.zeros(n_classes)
        distance_sum = np.zeros(n_classes)

        closest_idx = np.argmin(distance_matrix)
        assigned_class = labels[closest_idx] - 1
        class_count[assigned_class] += 1
        distance_sum[assigned_class] += distance_matrix[closest_idx]
        distance_matrix[closest_idx] = np.inf

        valid_classes = class_count > 0
        average_distance = np.divide(distance_sum, class_count, out=np.full_like(distance_sum, np.inf), where=valid_classes)
        return np.argmin(average_distance) + 1

    distances = pairwise_distance(ground_truth_coeff, predicted_coeff)
    # Get predictions and calculate accuracy
    predicted_labels = [classify_based_on_distance(dist_row, true_labels) for dist_row in distances]
    predicted_labels = np.array(predicted_labels, dtype=int)
    accuracy = (np.sum(true_labels == predicted_labels) / len(true_labels)) * 100
    return accuracy

if __name__ == "__main__":
    out_dir = '/home/aolivepe/Computer-Vision/HW10/results/PCA'

    labels = {}
    p_embed = {}
    accuracies = []
    
    # Prepare data and extract mean and normalize
    x_train, y_train = getData('/home/aolivepe/Computer-Vision/HW10/FaceRecognition/train')
    x_test, y_test = getData('/home/aolivepe/Computer-Vision/HW10/FaceRecognition/test')
    mean = np.mean(x_train, axis=1, keepdims=True)
    x_train = x_train - mean
    x_train = x_train / np.linalg.norm(x_train, axis=0)
    x_test = x_test - mean
    x_test = x_test / np.linalg.norm(x_test, axis=0)
    
    labels["train"] = y_train
    labels["test"] = y_test
    
    # Plot covariance matrix
    C, C_compr = getCovMatrix(x_train)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        C, 
        cmap='viridis', 
        cbar=True, 
        annot=False, 
        square=True
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'train_cov_matrix.jpg'))

    # Get embeddings, classify and test for different stages
    for stg in range(1, 25 + 1):
        real_c, pred_c = getCoeff(x_train, x_test, largeEig(x_train, C_compr, stg))
        
        p_embed[stg] = {}
        p_embed[stg]["train"] = real_c
        p_embed[stg]["test"] = pred_c
        
        accuracy = getAccuracy(y_test, real_c, pred_c)
        accuracies.append(accuracy)
        print(f"stg: {stg} -> acc = {accuracy}")

    with open(out_dir + '/accuracies.npy', 'wb') as f:
        np.save(f, accuracies)
        
    with open(out_dir + '/embeddings.pkl', 'wb') as f:
        pickle.dump(p_embed, f)
        
    with open(out_dir + '/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

