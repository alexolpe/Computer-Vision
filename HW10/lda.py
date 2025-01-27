import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import heapq
import seaborn as sns
import pickle

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

# Get within-class mean
def getClassMean(data, num_classes, samples_per_class):
    return np.hstack([
        np.mean(data[:, i * samples_per_class:(i + 1) * samples_per_class], axis=1).reshape(-1, 1)
        for i in range(num_classes)
    ])

# Calculate SW
def getSW(data, class_means, num_classes, samples_per_class):
    feature_dim = data.shape[0]
    SW = np.zeros((feature_dim, feature_dim))
    for i in range(num_classes):
        start_idx, end_idx = i * samples_per_class, (i + 1) * samples_per_class
        class_data = data[:, start_idx:end_idx]
        centered_data = class_data - class_means[:, [i]]
        SW += centered_data @ centered_data.T / samples_per_class
    return SW / num_classes

# Calculate SB
def getSB(class_means, global_mean, num_classes):
    mean_diff = class_means - global_mean
    return np.matmul(mean_diff, mean_diff.T) / num_classes

# Calculate all needed matrices
def getMatrices(data, out_dir):
    num_classes = 30
    samples_per_class = 21

    # Compute class and global means
    class_means = getClassMean(data, num_classes, samples_per_class)
    global_mean = np.mean(data, axis=1).reshape(-1, 1)

    # Calculate scatter matrices
    SB = getSB(class_means, global_mean, num_classes)
    SW = getSW(data, class_means, num_classes, samples_per_class)
    combined = np.matmul(np.linalg.pinv(SW), SB)

    # Plot results
    plotMatrix(SB, out_dir, 'SB.jpg', 'SB')
    plotMatrix(SW, out_dir, 'SW.jpg', 'SW')
    plotMatrix(combined, out_dir, 'SWSB.jpg', '$SW^{-1} SB$')

    return SB, SW, global_mean

# Function to plot matrices
def plotMatrix(C, out_dir, save_path, title):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        C, 
        cmap='viridis', 
        cbar=True, 
        annot=False, 
        square=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, save_path))

def normCol(matrix):
    return matrix / np.linalg.norm(matrix, axis=0)

# Calculate DB and Y
def largeEigVal(eigenvalues, eigenvectors, num_select):
    eigenvalues = np.abs(eigenvalues)
    largest_values = heapq.nlargest(num_select, eigenvalues)
    indices = [list(eigenvalues).index(val) for val in largest_values]
    return eigenvalues[indices], eigenvectors[:, indices]

# Calculate U
def smallEigVal(eigenvalues, eigenvectors, num_select):
    eigenvalues = np.abs(eigenvalues)
    smallest_values = heapq.nsmallest(num_select, eigenvalues)
    indices = [list(eigenvalues).index(val) for val in smallest_values]
    return eigenvalues[indices], eigenvectors[:, indices]

# Calculate all the needed matrices for the Yu and Yang approach
def largeEig(SB, SW, stg):
    w_SB, v_SB = np.linalg.eig(SB)
    v_SB = normCol(np.real(v_SB))
    DB, Y = largeEigVal(w_SB, v_SB, 25)
    
    Z = np.matmul(Y, np.diag(1 / np.sqrt(DB)))
    G = np.matmul(Z.T, np.matmul(SW, Z))
    
    w_G, v_G = np.linalg.eig(G)
    v_G = normCol(np.real(v_G))
    wg, U = smallEigVal(w_G, v_G, stg)
    return np.matmul(U.T, Z.T).T[:, :stg]

# Get real and predicted embeddings
def getCoeff(x_train, x_test, mean, large_eig):
    # Extract mean
    x_train = x_train - mean
    real_c = x_train.T @ large_eig
    # Extract mean
    x_test = x_test - mean
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
    out_dir = '/home/aolivepe/Computer-Vision/HW10/results/LDA'

    labels = {}
    p_embed = {}
    accuracies = []
    
    # Prepare data and normalize
    x_train, y_train = getData("/home/aolivepe/Computer-Vision/HW10/FaceRecognition/train")
    x_test, y_test = getData("/home/aolivepe/Computer-Vision/HW10/FaceRecognition/test")
    x_train = x_train / np.linalg.norm(x_train, axis=0)
    x_test = x_test / np.linalg.norm(x_test, axis=0)
    
    labels["train"] = y_train
    labels["test"] = y_test

    # Calculate needed matrices
    SB, SW, mean = getMatrices(x_train, out_dir)
    
    # Get embeddings, classify and test for different stages
    for stg in range(1, 25 + 1):
        real_c, pred_c = getCoeff(x_train, x_test, mean, largeEig(SB, SW, stg))
        
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
