import numpy as np
import matplotlib.pyplot as plt
import umap

# Accuracy PCA, LDA and Autoencoder
acc_PCA = np.load('/home/aolivepe/Computer-Vision/HW10/results/PCA/accuracies.npy')
acc_LDA = np.load('/home/aolivepe/Computer-Vision/HW10/results/LDA/accuracies.npy')
acc_auto = np.load('/home/aolivepe/Computer-Vision/HW10/results/Autoencoder/accuracies.npy')

plt.figure(figsize=(9, 7))
fig, ax = plt.subplots()
ax.plot(np.arange(1, 21, 1), acc_PCA[:20], label="PCA", marker='1', color="blue")
ax.plot(np.arange(1, 21, 1), acc_LDA[:20], label="LDA", marker='s', color="red")
ax.plot([3, 8, 16], acc_auto, label="Autoencoder" , marker="p", color="green")
ax.set_ylabel("Accuracy" )
ax.set_xlabel("p")
ax.set_xticks(np.arange(0, 21, 1))
ax.legend()
ax.grid()
plt.savefig("/home/aolivepe/Computer-Vision/HW10/results/accuracy_pca_lda_auto.jpg")

# UMAP PCA
emb_PCA = np.load('/home/aolivepe/Computer-Vision/HW10/results/PCA/embeddings.pkl', allow_pickle=True)
labels = np.load('/home/aolivepe/Computer-Vision/HW10/results/PCA/labels.pkl', allow_pickle=True)
num = 11
reducer = umap.UMAP()

all = np.concatenate((emb_PCA[num]["train"], emb_PCA[num]["test"]), axis=0)
print(all.shape)

embedding = reducer.fit_transform(all)
embedding.shape
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with 2 subplots in a row.

# First subplot: PCA train embeddings.
axes[0].scatter(embedding[:630, 0], embedding[:630, 1], c=labels["train"], cmap='Spectral', s=5)
axes[0].set_aspect('equal', 'datalim')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=labels["train"].min(), vmax=labels["train"].max())),
    ax=axes[0],
    boundaries=np.arange(31)-0.5
)
cbar.set_ticks(np.arange(30))
axes[0].set_title(f'Projection PCA train embeddings p = {num}', fontsize=15)

# Second subplot: PCA test embeddings.
axes[1].scatter(embedding[630:, 0], embedding[630:, 1], c=labels["train"], cmap='Spectral', s=5)
axes[1].set_aspect('equal', 'datalim')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=labels["train"].min(), vmax=labels["train"].max())),
    ax=axes[1],
    boundaries=np.arange(31)-0.5
)
cbar.set_ticks(np.arange(30))
axes[1].set_title(f'Projection PCA test embeddings p = {num}', fontsize=15)

# Save the combined figure.
plt.savefig(f"/home/aolivepe/Computer-Vision/HW10/results/PCA/umap_{num}.jpg")

# UMAP LDA
emb_LDA = np.load('/home/aolivepe/Computer-Vision/HW10/results/LDA/embeddings.pkl', allow_pickle=True)
labels = np.load('/home/aolivepe/Computer-Vision/HW10/results/LDA/labels.pkl', allow_pickle=True)
num = 11
reducer = umap.UMAP()

all = np.concatenate((emb_LDA[num]["train"], emb_LDA[num]["test"]), axis=0)
print(all.shape)

embedding = reducer.fit_transform(all)
embedding.shape
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with 2 subplots in a row.

# First subplot: LDA train embeddings.
axes[0].scatter(embedding[:630, 0], embedding[:630, 1], c=labels["train"], cmap='Spectral', s=5)
axes[0].set_aspect('equal', 'datalim')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=labels["train"].min(), vmax=labels["train"].max())),
    ax=axes[0],
    boundaries=np.arange(31)-0.5
)
cbar.set_ticks(np.arange(30))
axes[0].set_title(f'Projection LDA train embeddings p = {num}', fontsize=15)

# Second subplot: LDA test embeddings.
axes[1].scatter(embedding[630:, 0], embedding[630:, 1], c=labels["train"], cmap='Spectral', s=5)
axes[1].set_aspect('equal', 'datalim')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=labels["train"].min(), vmax=labels["train"].max())),
    ax=axes[1],
    boundaries=np.arange(31)-0.5
)
cbar.set_ticks(np.arange(30))
axes[1].set_title(f'Projection LDA test embeddings p = {num}', fontsize=15)

# Save the combined figure.
plt.savefig(f"/home/aolivepe/Computer-Vision/HW10/results/LDA/umap_{num}.jpg")

# UMAP Autoencoder
emb_Autoencoder = np.load('/home/aolivepe/Computer-Vision/HW10/results/Autoencoder/embeddings.pkl', allow_pickle=True)
labels = np.load('/home/aolivepe/Computer-Vision/HW10/results/Autoencoder/labels.pkl', allow_pickle=True)
num = 16
reducer = umap.UMAP()

all = np.concatenate((emb_Autoencoder[num]["train"], emb_Autoencoder[num]["test"]), axis=0)
print(all.shape)

embedding = reducer.fit_transform(all)
embedding.shape
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with 2 subplots in a row.

# First subplot: Autoencoder train embeddings.
axes[0].scatter(embedding[:630, 0], embedding[:630, 1], c=labels["train"], cmap='Spectral', s=5)
axes[0].set_aspect('equal', 'datalim')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=labels["train"].min(), vmax=labels["train"].max())),
    ax=axes[0],
    boundaries=np.arange(31)-0.5
)
cbar.set_ticks(np.arange(30))
axes[0].set_title(f'Projection Autoencoder train embeddings p = {num}', fontsize=15)

# Second subplot: Autoencoder test embeddings.
axes[1].scatter(embedding[630:, 0], embedding[630:, 1], c=labels["train"], cmap='Spectral', s=5)
axes[1].set_aspect('equal', 'datalim')
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=labels["train"].min(), vmax=labels["train"].max())),
    ax=axes[1],
    boundaries=np.arange(31)-0.5
)
cbar.set_ticks(np.arange(30))
axes[1].set_title(f'Projection Autoencoder test embeddings p = {num}', fontsize=15)

# Save the combined figure.
plt.savefig(f"/home/aolivepe/Computer-Vision/HW10/results/Autoencoder/umap_{num}.jpg")