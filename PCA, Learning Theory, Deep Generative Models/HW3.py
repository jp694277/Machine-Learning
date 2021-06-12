#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 09:10:26 2021

@author: wangziwen
"""

import gzip
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD # without center
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
PCA
"""

def load_data(data_folder):

    files = [
          'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz'
    ]
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))
        
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)

#load
train_images, train_labels = load_data('/Users/wangziwen/Documents/Graduation/1st/Statistic Learning/HW/HW3/')

#dim
print('dim:',train_images.shape)

#reshape data
train_data=train_images.reshape(60000,784)

f, ax = plt.subplots(5, 5)
a = [i for i in range(25)]
a = np.array(a).reshape(5,5)
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(train_data[a[i][j]].reshape(28,28))

plt.show()

pca = PCA(n_components = 333) #np.argmax(np.cumsum(pca.explained_variance_ratio_)>=0.99)+1
train_data_pca = pca.fit_transform(train_data)
tsvd = TruncatedSVD(n_components = 332) #np.argmax(np.cumsum(pca.explained_variance_ratio_)>=0.99)+1
train_data_tsvd = tsvd.fit_transform(train_data)

print('PCA dim:',train_data_pca.shape,'-> reduce from 784 to 333') #784 -> 333
print('PCA without centered dim:',train_data_tsvd.shape,'-> reduce from 784 to 332')

cumsum_pca = np.cumsum(pca.explained_variance_ratio_)
cumsum_tsvd = np.cumsum(tsvd.explained_variance_ratio_)

#1%, 5%, 20%, 50%, 80%, 95%, 99%
plt.figure(1, figsize=(6, 4))
plt.clf()
plt.plot([i+1 for i in range(333)],
         cumsum_pca,linewidth=2)
plt.plot(1, cumsum_pca[0], linestyle="None", marker="o",  color="black")
plt.plot(3, cumsum_pca[2], linestyle="None", marker="o",  color="black")
plt.plot(11, cumsum_pca[10], linestyle="None", marker="o",  color="black")
plt.plot(44, cumsum_pca[43], linestyle="None", marker="o",  color="black")
plt.plot(154, cumsum_pca[153], linestyle="None", marker="o",  color="black")
plt.plot(333, cumsum_pca[332], linestyle="None", marker="o",  color="black")
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.title('PCA')
plt.show()

#1%, 5%, 20%, 50%, 80%, 95%, 99%
plt.figure(1, figsize=(6, 4))
plt.clf()
plt.plot([i+1 for i in range(332)],
         cumsum_tsvd,linewidth=2)
plt.plot(1, cumsum_tsvd[0], linestyle="None", marker="o",  color="black")
plt.plot(3, cumsum_tsvd[2], linestyle="None", marker="o",  color="black")
plt.plot(11, cumsum_tsvd[10], linestyle="None", marker="o",  color="black")
plt.plot(44, cumsum_tsvd[43], linestyle="None", marker="o",  color="black")
plt.plot(154, cumsum_tsvd[153], linestyle="None", marker="o",  color="black")
plt.plot(332, cumsum_tsvd[331], linestyle="None", marker="o",  color="black")
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.title('PCA without centered')
plt.show()

f, ax = plt.subplots(1, 2)

ax[0].imshow(train_data[15,:].reshape(28,28))
ax[1].imshow(train_data[16,:].reshape(28,28))
ax[0].set_title('original image1')
ax[1].set_title('original image2')
plt.tight_layout() 
plt.show()

def train_pca(d):
    pca = PCA(n_components = d)
    train_data_pca = pca.fit_transform(train_data)
    return pca,train_data_pca

def tsvd_pca(d):
    tsvd = TruncatedSVD(n_components = d)
    train_data_tsvd = tsvd.fit_transform(train_data)
    return tsvd,train_data_tsvd

pca1,pca_1 = train_pca(1) #9.7%
pca2,pca_2 = train_pca(3) #22.97%
pca3,pca_3 = train_pca(11) #50.92%
pca4,pca_4 = train_pca(44) #80.33%
pca5,pca_5 = train_pca(154) #95.02%
pca6,pca_6 = train_pca(333) #99%

tsvd1,tsvd_1 = tsvd_pca(1) #5.8%
tsvd2,tsvd_2 = tsvd_pca(3) #21.15%
tsvd3,tsvd_3 = tsvd_pca(11) #50.77%
tsvd4,tsvd_4 = tsvd_pca(44) #80.31%
tsvd5,tsvd_5 = tsvd_pca(154) #95.02%
tsvd6,tsvd_6 = tsvd_pca(331) #99%

f, ax = plt.subplots(2, 3)

f.suptitle("PCA of image1", fontsize=12)

ax[0,0].imshow(pca1.inverse_transform(pca_1[15, :]).reshape(28, 28))
ax[0,1].imshow(pca2.inverse_transform(pca_2[15, :]).reshape(28, 28))
ax[0,2].imshow(pca3.inverse_transform(pca_3[15, :]).reshape(28, 28))
ax[1,0].imshow(pca4.inverse_transform(pca_4[15, :]).reshape(28, 28))
ax[1,1].imshow(pca5.inverse_transform(pca_5[15, :]).reshape(28, 28))
ax[1,2].imshow(pca6.inverse_transform(pca_6[15, :]).reshape(28, 28))

ax[0,0].set_title('1st component,9.7%',fontsize=7)
ax[0,1].set_title('1st~3rd components,22.97%',fontsize=7)
ax[0,2].set_title('1st~11th components,50.92%',fontsize=7)
ax[1,0].set_title('1st~44th components,80.33%',fontsize=7)
ax[1,1].set_title('1st~154th components,95.02%',fontsize=7)
ax[1,2].set_title('1st~333th components,99%',fontsize=7)

plt.tight_layout() 
plt.show()

f, ax = plt.subplots(2, 3)

f.suptitle("PCA without centered of image1", fontsize=12)

ax[0,0].imshow(tsvd1.inverse_transform(tsvd_1)[15, :].reshape(28, 28))
ax[0,1].imshow(tsvd2.inverse_transform(tsvd_2)[15, :].reshape(28, 28))
ax[0,2].imshow(tsvd3.inverse_transform(tsvd_3)[15, :].reshape(28, 28))
ax[1,0].imshow(tsvd4.inverse_transform(tsvd_4)[15, :].reshape(28, 28))
ax[1,1].imshow(tsvd5.inverse_transform(tsvd_5)[15, :].reshape(28, 28))
ax[1,2].imshow(tsvd6.inverse_transform(tsvd_6)[15, :].reshape(28, 28))

ax[0,0].set_title('1st component,5.8%',fontsize=7)
ax[0,1].set_title('1st~3rd components,21.15%',fontsize=7)
ax[0,2].set_title('1st~11th components,50.77%',fontsize=7)
ax[1,0].set_title('1st~44th components,80.31%',fontsize=7)
ax[1,1].set_title('1st~154th components,95.02%',fontsize=7)
ax[1,2].set_title('1st~331th components,99%',fontsize=7)

plt.tight_layout() 
plt.show()

f, ax = plt.subplots(2, 3)

f.suptitle("PCA of image2", fontsize=12)

ax[0,0].imshow(pca1.inverse_transform(pca_1[16, :]).reshape(28, 28))
ax[0,1].imshow(pca2.inverse_transform(pca_2[16, :]).reshape(28, 28))
ax[0,2].imshow(pca3.inverse_transform(pca_3[16, :]).reshape(28, 28))
ax[1,0].imshow(pca4.inverse_transform(pca_4[16, :]).reshape(28, 28))
ax[1,1].imshow(pca5.inverse_transform(pca_5[16, :]).reshape(28, 28))
ax[1,2].imshow(pca6.inverse_transform(pca_6[16, :]).reshape(28, 28))

ax[0,0].set_title('1st component,9.7%',fontsize=7)
ax[0,1].set_title('1st~3rd components,22.97%',fontsize=7)
ax[0,2].set_title('1st~11th components,50.92%',fontsize=7)
ax[1,0].set_title('1st~44th components,80.33%',fontsize=7)
ax[1,1].set_title('1st~154th components,95.02%',fontsize=7)
ax[1,2].set_title('1st~333th components,99%',fontsize=7)

plt.tight_layout() 
plt.show()

f, ax = plt.subplots(2, 3)

f.suptitle("PCA without centered of image2", fontsize=12)

ax[0,0].imshow(tsvd1.inverse_transform(tsvd_1)[16, :].reshape(28, 28))
ax[0,1].imshow(tsvd2.inverse_transform(tsvd_2)[16, :].reshape(28, 28))
ax[0,2].imshow(tsvd3.inverse_transform(tsvd_3)[16, :].reshape(28, 28))
ax[1,0].imshow(tsvd4.inverse_transform(tsvd_4)[16, :].reshape(28, 28))
ax[1,1].imshow(tsvd5.inverse_transform(tsvd_5)[16, :].reshape(28, 28))
ax[1,2].imshow(tsvd6.inverse_transform(tsvd_6)[16, :].reshape(28, 28))

ax[0,0].set_title('1st component,5.8%',fontsize=7)
ax[0,1].set_title('1st~3rd components,21.15%',fontsize=7)
ax[0,2].set_title('1st~11th components,50.77%',fontsize=7)
ax[1,0].set_title('1st~44th components,80.31%',fontsize=7)
ax[1,1].set_title('1st~154th components,95.02%',fontsize=7)
ax[1,2].set_title('1st~331th components,99%',fontsize=7)

plt.tight_layout() 
plt.show()

print('PCA eigenvector dim:',pca.components_.shape)
print('PCA without centered eigenvector dim:',tsvd.components_.shape)

print('PCA 100th eigenvectors - cumulative explained variance ratio:', cumsum_pca[99]*100)
print('PCA without centered 100th eigenvectors - cumulative explained variance ratio:', cumsum_tsvd[99]*100)

f, ax = plt.subplots(10, 10)

f.suptitle("PCA", fontsize=12)

a = [i for i in range(100)]
a = np.array(a).reshape(10,10)
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(pca.components_[a[i][j]].reshape(28,28))
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.show()

f, ax = plt.subplots(10, 10)

f.suptitle("PCA without centered", fontsize=12)

a = [i for i in range(100)]
a = np.array(a).reshape(10,10)
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(tsvd.components_[a[i][j]].reshape(28,28))
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])
plt.show()

f, ax = plt.subplots(2, 2)

ax[0,0].imshow(pca.components_[0,:].reshape(28,28))
ax[0,1].imshow(pca.components_[100,:].reshape(28,28))
ax[1,0].imshow(tsvd.components_[0,:].reshape(28,28))
ax[1,1].imshow(tsvd.components_[100,:].reshape(28,28))
ax[0,0].set_title('PCA - eigenvector 1st',fontsize = 8)
ax[0,1].set_title('PCA - eigenvector 100th',fontsize = 8)
ax[1,0].set_title('PCA without centered - eigenvector 1st',fontsize = 8)
ax[1,1].set_title('PCA without centered - eigenvector 100th',fontsize = 8)
plt.tight_layout() 
plt.show()

train_images[15,:] #稀疏矩陣
train_images[15,:] - np.mean(train_images[15,:])

"""
VAE
"""

"""
Bernoulli
"""
class BernoulliVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_units, dec_units):
        super(BernoulliVAE, self).__init__()
        # Encoder parameters
        self.linear_enc = nn.Linear(input_dim, enc_units)
        self.enc_mu = nn.Linear(enc_units, latent_dim)
        self.enc_logvar = nn.Linear(enc_units, latent_dim)

        # Distribution to sample for the reparameterization trick
        self.normal_dist = MultivariateNormal(torch.zeros(latent_dim),
                                              torch.eye(latent_dim))

        # Decoder parameters
        self.linear_dec = nn.Linear(latent_dim, dec_units)
        self.dec_mu = nn.Linear(dec_units, input_dim)

        # Reconstruction loss: binary cross-entropy
        self.criterion = nn.BCELoss(reduction='sum')

    def encode(self, x):
        # Obtain the parameters of the latent variable distribution
        h = torch.relu(self.linear_enc(x))
        mu_e = self.enc_mu(h)
        logvar_e = self.enc_logvar(h)

        # Get a latent variable sample with the reparameterization trick
        epsilon = self.normal_dist.sample((x.shape[0],))
        z = mu_e + torch.sqrt(torch.exp(logvar_e)) * epsilon

        return z, mu_e, logvar_e

    def decode(self, z):
        # Obtain the parameters of the observation distribution
        h = torch.relu(self.linear_dec(z))
        mu_d = torch.sigmoid(self.dec_mu(h))

        return mu_d

    def forward(self, x):
        """ Calculate the negative lower bound for the given input """
        z, mu_e, logvar_e = self.encode(x)
        mu_d = self.decode(z)
        neg_cross_entropy = self.criterion(mu_d, x)
        kl_div = -0.5* (1 + logvar_e - mu_e**2 - torch.exp(logvar_e)).sum()

        # Since the optimizer minimizes, we return the negative
        # of the lower bound that we need to maximize
        return neg_cross_entropy + kl_div
            
input_dim = 28 * 28
batch_size = 128
epochs = 5

dataset = datasets.MNIST('data/', transform=transforms.ToTensor(), download=True)
loader = DataLoader(dataset, batch_size, shuffle=True)
model = BernoulliVAE(input_dim, latent_dim=40, enc_units=200, dec_units=200) #40 dim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}')
    avg_loss = 0
    for i, (data, _) in enumerate(loader):
        model.zero_grad()
        # Reshape data so each image is an array with 784 elements
        data = data.view(-1, input_dim)

        loss = model(data)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()/len(dataset)

        if i % 100 == 0:
            # Print average loss per sample in batch
            batch_loss = loss.item()/len(data)
            print(f'\r[{i:d}/{len(loader):d}] batch loss: {batch_loss:.3f}',
                  end='', flush=True)

    print(f'\nAverage loss: {avg_loss:.6f}'.format(avg_loss))      
    

n_samples = 10
fig = plt.figure(figsize=(14, 3))
fig.suptitle('Apply VAE on MNIST, MLP = Bernoulli, dim of z = 40 (top row = orginal as bottom row = reconstruction)')
for i in range(n_samples):
    # Take a sample and view as mini-batch of size 1
    x = dataset[i][0].view(-1, input_dim)
    # Encode the observation
    z, mu_e, logvar_e = model.encode(x)
    # Get reconstruction
    x_d = model.decode(z)

    plt.subplot(2, n_samples, i + 1)
    plt.imshow(x.view(28, 28).data.numpy())
    plt.axis('off')
    plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(x_d.view(28, 28).data.numpy())
    plt.axis('off')

"""
Normal
"""
class GaussianVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_units, dec_units):
        super(GaussianVAE, self).__init__()
        # Encoder parameters
        self.linear_enc = nn.Linear(input_dim, enc_units)
        self.enc_mu = nn.Linear(enc_units, latent_dim)
        self.enc_logvar = nn.Linear(enc_units, latent_dim)

        # Distribution to sample for the reparameterization trick
        self.normal_dist = MultivariateNormal(torch.zeros(latent_dim),
                                              torch.eye(latent_dim))

        # Decoder parameters
        self.linear_dec = nn.Linear(latent_dim, dec_units)
        self.dec_mu = nn.Linear(dec_units, input_dim)

        # Reconstruction loss: binary cross-entropy
        self.criterion = nn.BCELoss(reduction='sum')

    def encode(self, x):
        # Obtain the parameters of the latent variable distribution
        h = torch.relu(self.linear_enc(x))
        mu_e = self.enc_mu(h)
        logvar_e = self.enc_logvar(h)

        # Get a latent variable sample with the reparameterization trick
        epsilon = self.normal_dist.sample((x.shape[0],))
        z = mu_e + torch.sqrt(torch.exp(logvar_e)) * epsilon

        return z, mu_e, logvar_e

    def decode(self, z):
        # Obtain the parameters of the observation distribution
        h = torch.relu(self.linear_dec(z))
        mu_d = torch.sigmoid(self.dec_mu(h))

        return mu_d

    
    def forward(self, x):
        """ Calculate the negative lower bound for the given input """
        z, mu_e, logvar_e = self.encode(x)
        mu_d = self.decode(z)
        neg_cross_entropy = self.criterion(mu_d, x)
        kl_div = -0.5* (1 + logvar_e - mu_e**2 - torch.exp(logvar_e)).sum()

        # Since the optimizer minimizes, we return the negative
        # of the lower bound that we need to maximize
        return neg_cross_entropy + kl_div
    
    def forward(self, x):
        z, mu_e, logvar_e = self.encode(x)
        mu_d = self.decode(z)
        BCE = reconstruction_function(mu_d, x)  # mse loss
        KLD_element = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD
    
reconstruction_function = nn.MSELoss(size_average=False)
input_dim = 28 * 28
batch_size = 128
epochs = 5

dataset = datasets.MNIST('data/', transform=transforms.ToTensor(), download=True)
loader = DataLoader(dataset, batch_size, shuffle=True)
model = GaussianVAE(input_dim, latent_dim=40, enc_units=200, dec_units=200) #40 dim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}')
    avg_loss = 0
    for i, (data, _) in enumerate(loader):
        model.zero_grad()
        # Reshape data so each image is an array with 784 elements
        data = data.view(-1, input_dim)

        loss = model(data)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()/len(dataset)

        if i % 100 == 0:
            # Print average loss per sample in batch
            batch_loss = loss.item()/len(data)
            print(f'\r[{i:d}/{len(loader):d}] batch loss: {batch_loss:.3f}',
                  end='', flush=True)

    print(f'\nAverage loss: {avg_loss:.6f}'.format(avg_loss))   
    
n_samples = 10
fig = plt.figure(figsize=(14, 3))
fig.suptitle('Apply VAE on MNIST, MLP = Gaussian, dim of z = 40 (top row = orginal as bottom row = reconstruction)')
for i in range(n_samples):
    # Take a sample and view as mini-batch of size 1
    x = dataset[i][0].view(-1, input_dim)
    # Encode the observation
    z, mu_e, logvar_e = model.encode(x)
    # Get reconstruction
    x_d = model.decode(z)

    plt.subplot(2, n_samples, i + 1)
    plt.imshow(x.view(28, 28).data.numpy())
    plt.axis('off')
    plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(x_d.view(28, 28).data.numpy())
    plt.axis('off')