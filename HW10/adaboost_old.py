import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import gzip, pickle, pickletools

def calculate_features(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    window_widths = np.arange(2, img.shape[1], 2)
    window_heights = np.arange(2, img.shape[0], 2)
    features = []

    for N in window_widths:
        img_padded = np.pad(img, ((0, 0), (int(N / 2), int(N / 2))), mode='constant')
        for ix in range(0, img.shape[0]):
            for jx in range(int(N / 2), img_padded.shape[1] - int(N / 2) + 1):
                neg_sums = np.sum(img_padded[ix, jx - int(N / 2):jx + 1].flatten()).astype(np.int32)
                pos_sums = np.sum(img_padded[ix, jx + 1:jx + int(N / 2) + 1].flatten()).astype(np.int32)
                features.append(pos_sums - neg_sums)

    for N in window_heights:
        img_padded = np.pad(img, ((int(N / 2), int(N / 2)), (0, 0)), mode='constant')
        for ix in range(int(N / 2), img_padded.shape[0] - int(N / 2) + 1):
            for jx in range(0, img.shape[1]):
                neg_sums = np.sum(img_padded[ix - int(N / 2):ix + 1, jx].flatten()).astype(np.int32)
                pos_sums = np.sum(img_padded[ix + 1:ix + int(N / 2) + 1, jx].flatten()).astype(np.int32)
                features.append(pos_sums - neg_sums)
    
    features = np.array(features)
    return features

def extract_features(classes, data_dir, result_dir, name):
    print("Constructing feature matrices for the dataset")
    for C in classes:
        save_path = result_dir + '/' + name + '_' + C + '.npy'
        if not os.path.exists(save_path):
            features = list()
            path_dir = data_dir + '/' + C
            for F in os.listdir(path_dir):
                print('Name ' + C + ' Images = ', len(features), end="\r")
                image = cv2.imread(path_dir + '/' + F)
                features.append(calculate_features(image))
                features = np.array(features)
                features = np.reshape(features, [len(features), -1])
                np.save(save_path, features)
                print("\n")

        else:
            print('Found feature matrices already in', save_path, end=' ')
            if C.upper() == "POSITIVE":
                data_pos_features = np.load(save_path)
                print(' === > ', data_pos_features.shape)
            elif C.upper() == "NEGATIVE":
                data_neg_features = np.load(save_path)
                print(' === > ', data_neg_features.shape)

    return data_pos_features, data_neg_features

def create_labels(pos_features, neg_features):
    combined_features = np.concatenate((pos_features, neg_features), axis=0)
    combined_labels = np.concatenate((np.ones((pos_features.shape[0], 1)),np.zeros((neg_features.shape[0], 1))),axis=0).astype(np.uint8)
    return combined_features, combined_labels.squeeze()

def create_weak_classifier(features, labels, weights):
    classifier_error = np.float64("inf")
    # iterate over each feature across images
    for F in range(0, features.shape[1]):
        feature_vector = features[:,F]
        indx = np.argsort(feature_vector)
        feature_vector_sorted, labels_sorted, weights_sorted = feature_vector[indx], labels[indx], weights[indx]
        positive_weights = np.zeros((features.shape[0], 1))
        negative_weights = np.zeros((features.shape[0], 1))
        positive_weights[labels_sorted==1, 0] = weights_sorted[labels_sorted == 1]
        negative_weights[labels_sorted == 0, 0] = weights_sorted[labels_sorted == 0]
        error_pol_1 = np.reshape(np.cumsum(positive_weights) + np.sum(negative_weights) - np.cumsum(negative_weights), [-1, 1])
        error_pol_2 = np.reshape(np.cumsum(negative_weights) + np.sum(positive_weights) - np.cumsum(positive_weights), [-1, 1])
        error = np.concatenate((error_pol_1, error_pol_2), axis = 1)
        min_idx = np.unravel_index(np.argmin(error), error.shape)
        min_err = error[min_idx]
        # weak classifier update
        if min_err < classifier_error :
            classifier_error = min_err
            feature_idx = F
            threshold = feature_vector_sorted[min_idx[0]]
            polarity = 1 if min_idx[1]==0 else 0
            classifications = feature_vector >= threshold if polarity == 1 else feature_vector<threshold
            classifier = [feature_idx, threshold, polarity, classifier_error, classifications]
    return classifier

def create_adaboost_cascade(features, labels, iterations_per_cascade, cascade_level):
    weights = np.concatenate((np.repeat(1/np.sum(labels==1), np.sum(labels==1)), np.repeat(1/np.sum(labels==0), np.sum(labels==0))), axis =0)
    best_classifier = None
    best_trust_factor = -np.float64("inf")
    for iter in range (iterations_per_cascade):
        weights = weights /np.sum(weights)# normalize the weights
        class_feature_idx, class_thresh, class_polarity, class_error, class_predicted_labels = create_weak_classifier(features, labels, weights)
        class_predicted_labels = np.where(class_predicted_labels ==0, -1, class_predicted_labels)
        epsilon_t = np.matmul(np.reshape(weights, [1, -1]), np.abs(class_predicted_labels - labels).reshape(-1 ,1)).squeeze() * 0.5
        trust_factor = np.log((1 - epsilon_t)/epsilon_t) * 0.5
        # update the weights
        weights = weights * np.exp(-trust_factor * labels * class_predicted_labels)
        FPR = np.sum(class_predicted_labels[np.sum(labels ==1):]==1)/np.sum(labels==0)
        FNR = 1 - (np.sum(class_predicted_labels[:np.sum(labels ==1)]==1)/np.sum(labels==1))
        print("Cascade Level = ", cascade_level +1, " Iteration = ", iter +1, " epsilon_t = ", round(epsilon_t, 4), "trust_factor = ", round(trust_factor, 4), " # Negatives=", np.sum(labels==0), " FPR = ", round(FPR*100, 4), "%FNR = ", round(FNR * 100, 4), "%")
        if trust_factor > best_trust_factor :
            best_trust_factor = trust_factor
            best_class_predictions = class_predicted_labels
            best_FPR = FPR
            best_FNR = FNR
            best_weak_classifier = [class_feature_idx, class_thresh, class_polarity, class_error, class_predicted_labels, FPR, FNR, epsilon_t, trust_factor]
    
    # revise the dataset for the next cascade layer
    # find out all the detections that have been classified as positive
    new_pos_features = features[:np.sum(labels==1), :]
    new_neg_features = features[np.sum(labels==1):, :]
    new_neg_features = new_neg_features[np.where(best_class_predictions[np.sum(labels ==1):] == 1), :][0]
    new_features, new_labels = create_labels(new_pos_features, new_neg_features)
    del new_pos_features, new_neg_features
    return new_features, new_labels, best_FPR, best_FNR, best_weak_classifier

def plot_cascade_adaboost_performance(FPRs, FNRs, iterations_per_cascade, num_cascades, save_dir, name, title=None):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    LINE_WIDTH = 3
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)  # Font size of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # Font size of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # Font size of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # Font size of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # Legend font size
    plt.rc('figure', titlesize=BIGGER_SIZE)  # Font size of the figure title
    fig = plt.figure()
    plt.plot(FPRs, '-*', label='FPR', linewidth=LINE_WIDTH)
    plt.plot(FNRs, '-d', label='FNR', linewidth=LINE_WIDTH)
    plt.legend()
    plt.xlabel('Number of Cascade Levels')
    plt.ylabel('Performance Metric')
    plt.xticks(
        np.arange(0, len(FPRs), 1),
        [str(ix) for ix in range(1, len(FPRs) + 1)]
    )
    plt.title(title)
    plt.tight_layout()

    # Save the figure in PDF and PNG formats
    save_base = save_dir + '/Adaboost_Performance_'
    save_suffix = f"{iterations_per_cascade}_{num_cascades}_{name}"
    plt.savefig(save_base + save_suffix + '.pdf')
    plt.savefig(save_base + save_suffix + '.png', dpi=600)

def inference(Best_Classifier_Per_Cascade_Level, features, labels):
    Final_classification_labels = np.ones((features.shape[0], 1))
    global_class_idx = np.arange(0, features.shape[0], 1).reshape(-1, 1)
    
    actual_positive_samples = np.sum(labels == 1)
    actual_negative_samples = np.sum(labels == 0)
    
    FPRs = []
    FNRs = []
    
    for ix in range(len(Best_Classifier_Per_Cascade_Level)):
        # Extract classifier parameters
        feature_ID = Best_Classifier_Per_Cascade_Level[str(ix + 1)][0]
        threshold = Best_Classifier_Per_Cascade_Level[str(ix + 1)][1]
        polarity = Best_Classifier_Per_Cascade_Level[str(ix + 1)][2]
        trust_factor = Best_Classifier_Per_Cascade_Level[str(ix + 1)][-1]
        
        # Extract and classify features
        feature_vector = features[:, feature_ID].reshape([-1, 1])
        classifications = feature_vector >= threshold if polarity == 1 else feature_vector < threshold
        classifications = np.where(classifications == 0, -1, classifications)
        
        # Calculate total and final classifications
        total_classifications = trust_factor * classifications
        final_classifications = total_classifications >= trust_factor
        
        # Calculate FPR and FNR
        if np.sum(labels == 0) == 0:
            FPR = 0
        else:
            FPR = np.sum(final_classifications[np.sum(labels == 1):] == 1) / np.sum(labels == 0)
        
        if np.sum(labels == 1) == 0:
            FNR = 0
        else:
            FNR = 1 - (np.sum(final_classifications[:np.sum(labels == 1)] == 1) / np.sum(labels == 1))
        
        # Update features, labels, and indices
        features = features[np.where(final_classifications == 1), :][0]
        labels = labels[np.where(final_classifications == 1)[0]]
        global_negative_class_idx = global_class_idx[np.where(final_classifications == 0), :][0]
        Final_classification_labels[global_negative_class_idx, 0] = -1
        global_class_idx = global_class_idx[np.where(final_classifications == 1), :][0]
        
        # Initialize Test_Predictions and store FPR and FNR
        if ix == 0:
            Test_Predictions = Final_classification_labels
            FPRs.append(FPR)
            FNRs.append(FNR)
    else:
        Test_Predictions = np.concatenate((Test_Predictions, Final_classification_labels), axis=1)
        FPRs.append(FPRs[-1] * FPR)
        FNRs.append(FNRs[-1] * FNR)

    return FPRs, FNRs, Test_Predictions

def main ():
    classes = ["positive","negative"]
    train_dir = "./CarDetection/train"
    test_dir = "./CarDetection/test"
    Result_dir = "./results/Adaboost"
    train = True
    num_cascades = 10
    iterations_per_cascade = 10
    current_FPR = 1
    current_FNR = 1
    FPRs = list ()
    FNRs = list ()
    tol = 1e-6
    Best_Classifier_Per_Cascade_Level = {}
    if not os.path.exists(Result_dir):
        os.makedirs(Result_dir)
    if train :
        train_pos_features, train_neg_features = extract_features(classes, train_dir, Result_dir, "train")
        train_features, train_labels = create_labels(train_pos_features, train_neg_features)
        # delete unnecessary duplicate data from memory
        del train_pos_features, train_neg_features
        for ix in range (num_cascades):
            # train_features, train_labels, best_FPR, best_FNR, best_weak_classifier = create_adaboost_cascade(train_features, train_labels, iterations_per_cascade, ix)
            
            weights = np.concatenate((np.repeat(1/np.sum(train_labels==1), np.sum(train_labels==1)), np.repeat(1/np.sum(train_labels==0), np.sum(train_labels==0))), axis =0)
            best_trust_factor = -np.float64("inf")
            for iter in range (iterations_per_cascade):
                weights = weights /np.sum(weights)# normalize the weights
                class_feature_idx, class_thresh, class_polarity, class_error, class_predicted_labels = create_weak_classifier(train_features, train_labels, weights)
                class_predicted_labels = np.where(class_predicted_labels ==0, -1, class_predicted_labels)
                epsilon_t = np.matmul(np.reshape(weights, [1, -1]), np.abs(class_predicted_labels - train_labels).reshape(-1 ,1)).squeeze() * 0.5
                trust_factor = np.log((1 - epsilon_t)/epsilon_t) * 0.5
                # update the weights
                weights = weights * np.exp(-trust_factor * train_labels * class_predicted_labels)
                FPR = np.sum(class_predicted_labels[np.sum(train_labels ==1):]==1)/np.sum(train_labels==0)
                FNR = 1 - (np.sum(class_predicted_labels[:np.sum(train_labels ==1)]==1)/np.sum(train_labels==1))
                print("Cascade Level = ", ix +1, " Iteration = ", iter +1, " epsilon_t = ", round(epsilon_t, 4), "trust_factor = ", round(trust_factor, 4), " # Negatives=", np.sum(train_labels==0), " FPR = ", round(FPR*100, 4), "%FNR = ", round(FNR * 100, 4), "%")
                if trust_factor > best_trust_factor :
                    best_trust_factor = trust_factor
                    best_class_predictions = class_predicted_labels
                    best_FPR = FPR
                    best_FNR = FNR
                    best_weak_classifier = [class_feature_idx, class_thresh, class_polarity, class_error, class_predicted_labels, FPR, FNR, epsilon_t, trust_factor]
            
            # revise the dataset for the next cascade layer
            # find out all the detections that have been classified as positive
            new_pos_features = train_features[:np.sum(train_labels==1), :]
            new_neg_features = train_features[np.sum(train_labels==1):, :]
            new_neg_features = new_neg_features[np.where(best_class_predictions[np.sum(train_labels ==1):] == 1), :][0]
            train_features, train_labels = create_labels(new_pos_features, new_neg_features)
            del new_pos_features, new_neg_features
            # return new_features, new_labels, best_FPR, best_FNR, best_weak_classifier
            Best_Classifier_Per_Cascade_Level[str(ix+1)] = best_weak_classifier
            current_FPR = current_FPR * best_FPR
            current_FNR = current_FNR * best_FNR
            FPRs.append(current_FPR)
            FNRs.append(current_FNR)
            print("Cumulative FPR = ", round(current_FPR, 4), " Cumulative FNR = ", round(current_FNR, 4))
            tol_track = ix+1
            if current_FPR <= tol :
                print ("Reached FPR Tolerance level of ", tol, " after cascade level = ", ix+1)
            if np.sum(train_labels ==0) == 0:
                print ("There is no longer any negative iabelled images remained after cascade level = ", ix+1)
                break
        with open(Result_dir +"/Adaboost_Performance_Metrics_"+ str(iterations_per_cascade)+"_"+str(num_cascades)+".pkl","wb") as f:
            pickle.dump(FPRs, f)
            pickle.dump(FNRs, f)
        with open(Result_dir +"/Adaboost_Best_Classifier_Per_Cascade_Level_"+str(iterations_per_cascade)+"_"+str(num_cascades)+".pkl","wb") as f:
            pickle.dump(Best_Classifier_Per_Cascade_Level, f)
        plot_cascade_adaboost_performance(np.array(FPRs), np.array(FNRs), iterations_per_cascade, num_cascades, Result_dir, "train", title = " Iter / Cascade = " + str(iterations_per_cascade)+" ,tolerance reached atcascade = "+ str(tol_track))
    else :
        # Performance on test data
        with open(Result_dir+"/Adaboost_Best_Classifier_Per_Cascade_Level_"+ str(iterations_per_cascade)+"_"+str(num_cascades)+".pkl","rb ") as f:
            Best_Classifier_Per_Cascade_Level = pickle.load(f)
        test_pos_features, test_neg_features = extract_features(classes, test_dir, Result_dir,"test")
        test_features, test_labels = create_labels(test_pos_features, test_neg_features)
        FPRs, FNRs, Test_Predictions = inference(Best_Classifier_Per_Cascade_Level, test_features, test_labels)
        plot_cascade_adaboost_performance(np.array(FPRs), np.array(FNRs), iterations_per_cascade, num_cascades, Result_dir, "test")
main()
