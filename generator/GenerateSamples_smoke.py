# -*- coding: utf-8 -*-

import collections
import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import os
print(torch.version.cuda) #10.1
import time
t0 = time.time()


##############################################################################
"""args for AE"""

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 3#1    # number of channels in the input data 

args['n_z'] = 600 #300     # number of dimensions in latent space. 


args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 200       # how many epochs to run for
args['batch_size'] = 1   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'mnist'  #'fmnist' # specify which dataset to use


##############################################################################



## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            
            nn.BatchNorm2d(self.dim_h * 8), # 40 X 8 = 320
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(self.dim_h * 8, self.dim_h * 8, 7, 7, 0, bias=False), # 640x480
            nn.Conv2d(self.dim_h * 8, self.dim_h * 8, 3, 3, 0, bias=False), # 640x480

            nn.BatchNorm2d(self.dim_h * 8), # 40 X 8 = 320
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),     
            )

        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
        

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 32),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 4, 16),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 16),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h * 2, 8),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, kernel_size=9, stride=7, padding=0, output_padding=0),
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False),
            # #nn.Sigmoid())
            nn.Tanh()
            )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x


##############################################################################

def biased_get_class1(c):

    xbeg = [img for img, label in dataset if label == c]
    ybeg = [label for _, label in dataset if label == c]

    return xbeg, ybeg


def G_SM1(X, y,n_to_sample,cl):

    
    # fitting the model
    n_neigh = 1 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

#############################################################################
np.printoptions(precision=5,suppress=True)

# Define the path to your dataset
data_path = './data/smoke100k/'





# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the images to a fixed size
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])
# transform = transforms.ToPILImage()

# Create an instance of the ImageFolder dataset
dataset = ImageFolder(root=data_path, transform=transform)
print("dataset", dataset)
print("dataset category name-idx", dataset.class_to_idx.items())
print("dataset category name", dataset.class_to_idx.keys())
print("dataset category idx", dataset.class_to_idx.values())
print("categories", len(dataset.class_to_idx.keys()))


# Print the class labels
class_labels = dataset.classes
print("Class Labels:", class_labels)

# Create a DataLoader for efficient batch processing
batch_size = 2
eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#path on the computer where the models are stored
modpth = './smoke/models/crs5/'

encf = []
decf = []
# for p in range(5):
for p in range(1):

    enc = './trained_models/mod3/bst_enc.pth'
    dec = './trained_models/mod3/bst_dec.pth'
    
    encf.append(enc)
    decf.append(dec)


# for m in range(5):
for m in range(1):
    print(m)
    
    #generate some images 
    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    path_enc = encf[m]
    path_dec = decf[m]
    print("path_enc", path_enc)
    print("path_dec", path_dec)

    encoder = Encoder(args)
    encoder.load_state_dict(torch.load(path_enc))
    encoder = encoder.to(device)

    decoder = Decoder(args)
    decoder.load_state_dict(torch.load(path_dec))
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    #imbal = [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]
    # imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    imbal = [1001, 1000]


    resx = []
    resy = []


    # for i in range(1,10):
    for i in range(1,2): # for imbal = []
        for j, (images,labs) in enumerate(eval_loader):
            
            print("images shape", images.shape)
            print("labs shape", labs.shape)
            xclass, yclass = images, labs
                
            #encode xclass to feature space
            xclass = xclass.to(device)
            xclass = encoder(xclass)
                
            xclass = xclass.detach().cpu().numpy()
            n = imbal[0] - imbal[i]
            xsamp, ysamp = G_SM1(xclass,yclass,n,i)
            ysamp = np.array(ysamp)
        
            """to generate samples for resnet"""   
            xsamp = torch.Tensor(xsamp)
            xsamp = xsamp.to(device)
            ximg = decoder(xsamp)
            img_save = torch.squeeze(ximg)
            print("img_save shape", img_save.shape)
            print("input image path", dataset.imgs[j][0])
            save_name = os.path.basename(dataset.imgs[j][0])
            vutils.save_image(img_save, "./smote_gen/smoke100k/{}".format(save_name))


t1 = time.time()
print('final time(min): {:.2f}'.format((t1 - t0)/60))


