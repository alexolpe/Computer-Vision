import os
import pickle
import numpy as np
import torch
from torch import nn, optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DataBuilder(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_list = [f for f in os.listdir(path) if f.endswith('.png')]
        self.label_list = [int(f.split('_')[0]) for f in self.image_list]
        self.len = len(self.image_list)
        self.aug = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        fn = os.path.join(self.path, self.image_list[index])
        x = Image.open(fn).convert('RGB')
        x = self.aug(x)
        return {'x': x, 'y': self.label_list[index]}

    def __len__(self):
        return self.len


class Autoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, encoded_space_dim * 2)
        )
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 4 * 4 * 64),
            nn.LeakyReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu, logvar = x[:, :self.encoded_space_dim], x[:, self.encoded_space_dim:]
        return mu, logvar

    def decode(self, z):
        x = self.decoder_lin(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


class VaeLoss(nn.Module):
    def __init__(self):
        super(VaeLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, xhat, x, mu, logvar):
        loss_MSE = self.mse_loss(xhat, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad()
        mu, logvar = model.encode(data['x'])
        z = model.reparameterize(mu, logvar)
        xhat = model.decode(z)
        loss = vae_loss(xhat, data['x'], mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(trainloader.dataset)))


##################################
# Change these
#p = 3  # [3, 8, 16]
training = False
TRAIN_DATA_PATH = '/home/aolivepe/Computer-Vision/HW10/FaceRecognition/train'
EVAL_DATA_PATH = '/home/aolivepe/Computer-Vision/HW10/FaceRecognition/test'
OUT_PATH = '/home/aolivepe/Computer-Vision/HW10/results/Autoencoder'

##################################
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

labels = {}
p_embed = {}
accuracies = []
    
# Iterate over the 3 values of P
for p in [3, 8, 16]:
    LOAD_PATH = f'/home/aolivepe/Computer-Vision/HW10/weights/model_{p}.pt'
    model = Autoencoder(p)

    if training:
        epochs = 100
        log_interval = 1
        trainloader = DataLoader(
            dataset=DataBuilder(TRAIN_DATA_PATH),
            batch_size=12,
            shuffle=True,
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        vae_loss = VaeLoss()
        for epoch in range(1, epochs + 1):
            train(epoch)
        torch.save(model.state_dict(), os.path.join(OUT_PATH, f'model_{p}.pt'))
    else:
        trainloader = DataLoader(
            dataset=DataBuilder(TRAIN_DATA_PATH),
            batch_size=1,
        )
        model.load_state_dict(torch.load(LOAD_PATH))
        model.eval()

        X_train, y_train = [], []
        for batch_idx, data in enumerate(trainloader):
            mu, logvar = model.encode(data['x'])
            z = mu.detach().cpu().numpy().flatten()
            X_train.append(z)
            y_train.append(data['y'].item())
        X_train = np.stack(X_train)
        y_train = np.array(y_train)

        testloader = DataLoader(
            dataset=DataBuilder(EVAL_DATA_PATH),
            batch_size=1,
        )
        X_test, y_test = [], []
        for batch_idx, data in enumerate(testloader):
            mu, logvar = model.encode(data['x'])
            z = mu.detach().cpu().numpy().flatten()
            X_test.append(z)
            y_test.append(data['y'].item())
        X_test = np.stack(X_test)
        y_test = np.array(y_test)

        ##################################
        # Your code starts here
        
        labels["train"] = y_train
        labels["test"] = y_test
    
        p_embed[p] = {}
        p_embed[p]["train"] = X_train
        p_embed[p]["test"] = X_test
        
        # Get accuracy
        accuracy = getAccuracy(y_test, X_train, X_test)
        print(f"P: {p} -> acc = {accuracy}")
        accuracies.append(accuracy)

# Save information
with open(OUT_PATH + '/accuracies.npy', 'wb') as f:
    np.save(f, accuracies)
    
with open(OUT_PATH + '/embeddings.pkl', 'wb') as f:
    pickle.dump(p_embed, f)
    
with open(OUT_PATH + '/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

        ##################################
