import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import autoencoder

def readImages(root_dir, sub_dir):
    # saves images to npz format if reading for the first time, else loads the images from the npz file
    if not os.path.exists(f'{root_dir}_{sub_dir}.npz'):
        img_list = []
        label_list = []
        img_path_list = glob.glob(os.path.join(root_dir, sub_dir, f'*'))
        for pth in img_path_list:
            # read image and append to array
            im_flat = cv2.imread(pth, 0).flatten()
            img_list.append(im_flat)
            # create array for labels
            f_name = os.path.split(pth)[-1]
            f_name = os.path.splitext(f_name)[0]
            c_num = int(f_name.split('_' )[0])
            label_list.append(c_num)
        np.savez_compressed(f'{root_dir}_{sub_dir}' , x=np.array(img_list), y=np.array(label_list))
    imgs = np.load(f'{root_dir}_{sub_dir}.npz' , allow_pickle=True)
    return imgs['x'], imgs['y']

def normalize(x):
    x = np.divide(x, np.expand_dims(np.linalg.norm(x, axis=1), axis=-1))
    return x

def calcPCA(x, p=None):
    x = np.transpose(x) # arranged as column vectors
    m = np.mean(x, axis=1, keepdims=True)
    x = x - m
    # using computational trick
    cov = np.dot(np.transpose(x), x) # calc X'X
    _, _, ut = np.linalg.svd(cov) # eigen vecs of X'X
    # multipy by X to get eigenvecs of XX'
    w = np.dot(x, ut)
    w = normalize(w)
    # retain first p eigen vecs
    w_p = w[:, :p]
    return w_p, m

def calcLDA(x, y, p):
    num_clss = np.max(y)
    dat_per_class = np.zeros(num_clss)
    for val in y: dat_per_class[val - 1] += 1 # count number of samples in each class
    cl_ms = np.zeros((num_clss, x.shape[1]))
    # compute per class means
    for data, lbl in zip(x, y): cl_ms[lbl - 1] += data
    cl_ms = np.transpose(cl_ms / np.expand_dims(dat_per_class, axis=1))
    x = np.transpose(x)
    m = np.mean(cl_ms, axis=1, keepdims=True) # global mean
    m_i = cl_ms - m
    x_i = x - m
    # -----yu-yang method-----#
    # using computational trick
    sb = np.dot(np.transpose(m_i), m_i)
    _, d, ut = np.linalg.svd(sb) # eigen vecs of X'X
    # multipy by X to get eigenvecs of XX'
    w = np.dot(m_i, ut)
    w = normalize(w)
    # remove eigen-vec corresponding to eigenval closest to 0; last one in this case
    y = w[:, :-1]
    D = d[:-1]
    D_b = np.diag(D)
    Z = np.dot(y, np.sqrt(np.linalg.inv(D_b)))
    zt_x = np.dot(np.transpose(Z), x_i) ##
    sw = np.dot(zt_x, np.transpose(zt_x))
    _, d, u_t = np.linalg.svd(sw)
    # multipy by X to get eigenvecs of XX'
    w = np.dot(Z, u_t)
    w = normalize(w)
    # retain first p eigen vecs
    w_p = w[:, :p]
    return w_p, m

# 1 nearest neighbor classifier
def oneNN(ft_space, p_x_test, y_train):
    p_x_test = np.expand_dims(p_x_test, axis=1)
    ft = np.transpose(ft_space)
    pxt = np.transpose(p_x_test)
    # calculate L2 distance between test vec and trained feature subspace
    dist = np.sqrt(np.linalg.norm(ft - pxt, axis=1))
    pred_label = y_train[np.argmin(dist)]
    return pred_label

# TASK 1: PCA and LDA Features Classification
def classifier(x_train, x_test, y_train, y_test, p):
    # calculate principal components and linear discriminants
    W_p_PCA, m = calcPCA(x_train, p)
    W_p_LDA, _ = calcLDA(x_train, y_train, p)
    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)
    # project onto respective subspaces
    PCA_space = np.dot(np.transpose(W_p_PCA), x_train - m)
    LDA_space = np.dot(np.transpose(W_p_LDA), x_train - m)
    num_test_samples = len(y_test)
    x_testm = x_test - m
    pred_lbls_PCA = []
    pred_lbls_LDA = []
    # make predictions using nearest neighbor classifier
    for i in range(num_test_samples):
        PCA_x_test = np.dot(np.transpose(W_p_PCA), x_testm[:, i])
        LDA_x_test = np.dot(np.transpose(W_p_LDA), x_testm[:, i])
        pred_PCA = oneNN(PCA_space, PCA_x_test, y_train)
        pred_LDA = oneNN(LDA_space, LDA_x_test, y_train)
        pred_lbls_PCA.append(pred_PCA)
        pred_lbls_LDA.append(pred_LDA)
    # calculate classification accuracy
    accuracy_PCA = np.mean(np.equal(np.array(pred_lbls_PCA), y_test)) * 100
    accuracy_LDA = np.mean(np.equal(np.array(pred_lbls_LDA), y_test)) * 100
    return accuracy_PCA, accuracy_LDA

# TASK 2: Autoencoder Features Classification
def autoenc_classifier(X_train, Y_train, X_test, Y_test):
    pred_lbls = []
    num_test_samples = len(Y_test)
    for i in range(num_test_samples):
        pred = oneNN(np.transpose(X_train), X_test[i], Y_train)
        pred_lbls.append(pred)
    accuracy = np.mean(np.equal(np.array(pred_lbls), Y_test)) * 100
    return accuracy

def main():
    root_dir = 'FaceRecognition'
    x_train, y_train = readImages(root_dir, 'train' )
    x_test, y_test = readImages(root_dir, 'test' )
    x_train, x_test = normalize(x_train), normalize(x_test)
    acc_PCA, acc_LDA, acc_auto = [], [], []
    # evaluate PCA and LDA classifiers for different subspace dimensions
    for p in range(1, 21):
        pca_acc, lda_acc = classifier(x_train, x_test, y_train, y_test, p)
        acc_PCA.append(pca_acc)
        acc_LDA.append(lda_acc)
    # evaluate autoencoder classifier for different subspace dimensions
    # for p in [3, 8, 16]:
    #     X_train, Y_train, X_test, Y_test = autoencoder.auto_encoder(p)
    #     auto_acc = autoenc_classifier(X_train, Y_train, X_test, Y_test)
    #     acc_auto.append(auto_acc)
        
    # plot accuracy vs subspace dimension
    rcParams['figure.dpi' ] = 300
    plt.figure(figsize=(9, 7))
    fig, ax = plt.subplots()
    ps = np.arange(1, 21, 1)
    ax.plot(ps, acc_PCA, label="PCA" , marker='x' )
    ax.plot(ps, acc_LDA, label="LDA" , marker='o' )
    # ax.plot([3, 8, 16], acc_auto, label="Autoencoder" , marker="*" )
    ax.set_title("Accuracy of Different Dimensionality Reduction Methods vs P" )
    ax.set_ylabel("Accuracy %" )
    ax.set_xlabel("Feature Dimension (P)" )
    ax.set_xticks(np.arange(0, 22, 1))
    ax.legend()
    plt.savefig("/home/aolivepe/Computer-Vision/HW10/results/result.jpg")

if __name__ == '__main__' :
    main()
