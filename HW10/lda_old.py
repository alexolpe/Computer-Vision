import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import linalg
import scipy.io
import cv2, os
import heapq
from collections import Counter
import pickle

def read_data(data_dir, num_classes, image_per_class):
    labels = list()
    for ix in range(1, num_classes + 1):
        for jx in range(1, image_per_class + 1):
            class_id = str(ix) if ix >= 10 else '0' + str(ix) # He does this to deal with 00, 01, 02, 03, 04 ...
            image_id = str(jx) if jx >= 10 else '0' + str(jx)
            image = cv2.imread(data_dir + '/' + class_id + '_' + image_id + '.png')
            image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if ix == 1 and jx == 1:
                image_shape = image.shape
            image = np.reshape(image.flatten(), [-1, 1])
            if ix == 1 and jx == 1:
                image_set = image
            else:
                image_set = np.concatenate((image_set, image), axis=1)
            labels.append(ix)
    labels = np.array(labels).squeeze()
    return image_set, labels, image_shape

def center_data(data, save_dir):
    mean_data = np.mean(data, axis=1)
    with open(save_dir + '/image_mean.npy', 'wb') as f:
        np.save(f, mean_data)
    data = data - np.reshape(mean_data, [-1, 1])
    return data

def normalize_data(data):
    data_norm = np.linalg.norm(data, axis=0)
    data_norm = np.reshape(data_norm, [1, -1])
    data_norm = np.tile(data_norm, (data.shape[0], 1))
    data = np.divide(data, data_norm)
    return data

def lda_mean_calculation(data, num_classes, image_per_class, save_dir):
    for ix in range(0, num_classes):
        class_data = data[:, ix*image_per_class:(ix+1)*image_per_class]
        mean_class_data = np.mean(class_data, axis=1).reshape([-1 ,1])
        if ix == 0:
            within_class_mean = mean_class_data
        else :
            within_class_mean = np.concatenate((within_class_mean, mean_class_data), axis=1)
    global_mean = np.mean(data, axis=1).reshape([-1, 1])
    with open(save_dir+"/lda_means.npy", "wb") as f:
        pickle.dump(within_class_mean, f)
        pickle.dump(global_mean, f)
    return within_class_mean, global_mean

def calculate_scatter_matrices(
    data, within_class_mean, global_mean, num_classes, image_per_class, save_dir, mode='calc'):
    if mode.upper() == "CALC":
        M = within_class_mean - global_mean
        SB = np.matmul(M, M.T) / within_class_mean.shape[1]
        SW = np.zeros(SB.shape)
        
        for ix in range(0, num_classes):
            class_data = data[:, ix * image_per_class:(ix + 1) * image_per_class]
            M = class_data - np.reshape(within_class_mean[:, ix], [-1, 1])
            SW += np.matmul(M, M.T) / float(image_per_class)
        
        SW = SW / float(num_classes)
        combined_scatter_matrix = np.matmul(np.linalg.pinv(SW), SB)
        
        with open(save_dir + '/scatter_matrices_.pkl', 'wb') as f:
            pickle.dump(SW, f)
            pickle.dump(SB, f)
            pickle.dump(combined_scatter_matrix, f)
    
    elif mode.upper() == "LOAD":
        print("Loading Scatter Matrices")
        with open(save_dir + '/scatter_matrices_.pkl', 'rb') as f:
            SB = pickle.load(f)
            SW = pickle.load(f)
            combined_scatter_matrix = pickle.load(f)
    else:
        print("Please specify a correct mode of operation, either 'calc' or 'load'")
    
    return SB, SW, combined_scatter_matrix


def covariance_matrix(data_set, save_dir, mode='calc'):
    if mode.upper() == "CALC":
        C = np.matmul(data_set, data_set.T)
        C_compressed = np.matmul(data_set.T, data_set)
        with open(save_dir + '/train_data_covariance_matrix.npy', 'wb') as f:
            np.save(f, C)
            np.save(f, C_compressed)
    elif mode.upper() == "LOAD":
        with open(save_dir + '/train_data_covariance_matrix.npy', 'rb') as f:
            C = np.load(f)
            C_compressed = np.load(f)
    return C, C_compressed

def plot_covariance_matrix(C_matrix, save_dir, save_path):
    fig = plt.figure()
    # C_matrix = C_matrix / np.max(C_matrix)
    plt.matshow(C_matrix)
    plt.colorbar()
    plt.savefig(save_dir + '/' + save_path)


def plot_eigenvectors(eigs, K, save_dir):
    plt.figure()
    plt.plot(eigs[:, :K])
    plt.legend(['E' + str(i) for i in range(1, K + 1)])
    plt.savefig(save_dir + '/eigenvectors_K_' + str(K) + '.png')


def plot_eigen_faces(eigs, save_dir, image_shape, num_faces):
    count = 0
    num_rows = int(np.sqrt(num_faces))
    num_cols = int(num_faces / float(num_rows))
    eig_f = np.zeros((image_shape[0] * num_rows, image_shape[1] * num_cols))
    for i in range(0, num_faces):
        row = int(np.floor((i) / num_cols)) * image_shape[0]
        col = count * image_shape[1]
        eig_f[row:row + image_shape[0], col:col + image_shape[1]] = np.reshape(eigs[:, i], list(image_shape))
        count += 1
        if count >= num_cols:
            count = 0
    plt.figure()
    plt.imshow(eig_f)
    plt.savefig(save_dir + '/Eigen_Faces.png')


def calculate_K_large_eigenvectors(SB, SW, combined_scatter_matrix, K, save_dir, lda_mode_params, mode='calc'):
    if mode.upper() == "CALC":
        KY = lda_mode_params[0]
        [w, v] = np.linalg.eig(SB)
        v = np.real(v)
        v = v / np.linalg.norm(v, axis=0)
        w = np.abs(w)
        ws = heapq.nlargest(KY, w)
        ind = [list(w).index(i) for i in ws]
        Y = v[:, ind]
        DB = w[ind]

        Z = np.matmul(Y, np.diag(1 / np.sqrt(DB)))
        G = np.matmul(Z.T, np.matmul(SW, Z))
        [wg, vg] = np.linalg.eig(G)
        vg = np.real(vg)
        vg = vg / np.linalg.norm(vg, axis=0)
        wg = np.abs(wg)
        wgs = heapq.nsmallest(K, wg)
        indg = [list(wg).index(i) for i in wgs]
        U = vg[:, indg]
        Q = np.matmul(U.T, Z.T).T

        with open(save_dir + '/eigenvectors.npy', 'wb') as f:
            pickle.dump(Q, f)

    elif mode.upper() == "LOAD":
        with open(save_dir + '/eigenvectors.npy', 'rb') as f:
            Q = np.load(f)
    # Return top K projections Q[:, :K]
    # I think that the v and w we are returning are not correct (nothing to do with the Q) because they are not equally sorted as Q. Nevertheless they are not used in the code
    return Q[:, :K]


def get_low_dimensional_projections(v, Q, data, save_dir):
    ak = []
    bk = []
    ck = []
    for i in range(0, data.shape[1]):
        ak.append(np.dot(data[:, i], Q[:, 0]))
        bk.append(np.dot(data[:, i], Q[:, 1]))
        ck.append(np.dot(data[:, i], Q[:, 2]))
    ak, bk, ck = np.array(ak), np.array(bk), np.array(ck)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ak, bk, ck)
    ax.set_xlabel('eig_0')
    ax.set_ylabel('eig_1')
    ax.set_zlabel('eig_2')
    plt.savefig(save_dir + '/projections_along_first_three_eig.png')
    return ak, bk, ck


def get_gt_coeffs(data, Q, K, image_shape, save_dir, p_embed):
    coeff = np.matmul(data.T, Q)
    p_embed[K] = {}
    p_embed[K]["train"] = coeff
    with open(save_dir + '/coefficients.npy', 'wb') as f:
        np.save(f, coeff)
    return coeff, p_embed


def get_prediction(data, Q, K, image_shape, save_dir, p_embed):
    with open(save_dir+"/lda_means.pkl", "rb") as f:
        within_class_mean = pickle.load(f)
        image_mean = np.reshape(pickle.load(f), [-1 ,1])
    with open(save_dir + '/coefficients.npy', 'rb') as f:
        coeff_gt = np.load(f)
    # Extract the mean from the training data to the testing data
    data = data - image_mean
    # data = normalize_data(data)
    coeff_pred = np.matmul(data.T, Q)
    p_embed[K]["test"] = coeff_pred
    distance = np.zeros((data.shape[1], coeff_gt.shape[0]))
    for ix in range(0, data.shape[1]):
        distance[ix, :] = np.sqrt(np.sum(np.square(coeff_gt - coeff_pred[ix, :]), axis=1)).T
    return distance, p_embed

def calculate_accuracy(y_true, distance, image_per_class, num_classes, method="L2", params=None):
    if method.upper() == "L2":
        match_ind = np.argmin(distance, axis=1) #+ 1  # +1 for indices starting from 1
        # print("match_ind: ", match_ind)
        # print("len(y_true): ", len(y_true))
        y_pred = y_true[match_ind]
        y_pred = y_pred.astype(int).squeeze()
    elif method.upper() == "KNN":
        num_neighbours = params[0]
        y_pred = list()
        for im in range(0, distance.shape[0]):
            dist = distance[im, :]
            class_hits = np.zeros((num_classes))
            class_dist = np.zeros((num_classes))
            for ix in range(0, num_neighbours):
                min_idx = np.argmin(dist)
                min_label = y_true[min_idx] - 1
                class_hits[min_label] += 1
                class_dist[min_label] += dist[min_idx]
                dist[min_idx] = float('inf')
            max_hits = np.max(class_hits)
            class_avg_dist = class_dist / max_hits
            class_avg_dist = np.where(class_avg_dist == 0, np.inf, class_avg_dist)
            class_id = np.argmin(class_avg_dist) + 1
            y_pred.append(class_id)
        y_pred = np.array(y_pred).astype(int).squeeze()
    match = np.where(y_true == y_pred, 1, 0)
    correct_labels = np.sum(match)
    accuracy = (correct_labels / y_true.shape[0]) * 100
    return accuracy


def main():
    train_dir = './FaceRecognition/train'
    test_dir = './FaceRecognition/test'
    save_dir = './results/LDA'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_classes = 30
    train_image_per_class = 21
    test_image_per_class = 21
    classifier = "KNN"  # Options: KNN, L2
    params = [1]
    Kmax = 25
    lda_mode_params = [11]

    labels = {}
    X_train, Y_train, image_shape = read_data(train_dir, num_classes, train_image_per_class)
    X_test, Y_test, _ = read_data(test_dir, num_classes, test_image_per_class)
    labels["train"] = Y_train
    labels["test"] = Y_test
    X_train = center_data(X_train, save_dir)
    X_train = normalize_data(X_train)

    print('Train data =', X_train.shape)
    print('Test data =', X_test.shape)

    within_class_mean, global_mean = lda_mean_calculation(X_train, num_classes, train_image_per_class, save_dir)
    SB, SW, combined_scatter_matrix = calculate_scatter_matrices(X_train, within_class_mean, global_mean, num_classes, train_image_per_class, save_dir, mode="calc")
    print("SB = ", SB.shape)
    print("SW = ", SW.shape)
    plot_covariance_matrix(SB, save_dir,"train_data_SB.png")
    plot_covariance_matrix(SW, save_dir,"train_data_SW.png")
    plot_covariance_matrix(combined_scatter_matrix, save_dir,"train_data_combined_scatter_matrix.png")

    p_embed = {}
    Accuracies = list()
    for K in range(1, Kmax + 1, 1):
        Q = calculate_K_large_eigenvectors(SB, SW, combined_scatter_matrix, K, save_dir, lda_mode_params, mode = "calc")
        coeff, p_embed = get_gt_coeffs(X_train-global_mean, Q, K, image_shape, save_dir, p_embed)
        distance, p_embed = get_prediction(X_test, Q, K, image_shape, save_dir, p_embed)
        accuracy = calculate_accuracy(Y_test, distance, test_image_per_class, num_classes, method = classifier, params = params)
        Accuracies.append(accuracy)

    with open(save_dir + '/LDA_Accuracies.npy', 'wb') as f:
        np.save(f, Accuracies)
        
    with open(save_dir + '/LDA_embeddings.pkl', 'wb') as f:
        pickle.dump(p_embed, f)
        
    with open(save_dir + '/LDA_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

main()
