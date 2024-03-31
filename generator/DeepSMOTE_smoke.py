
import collections
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

print(torch.version.cuda) #10.1
t3 = time.time()
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
args['batch_size'] = 100   # batch size for SGD
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
"""set models, loss functions"""
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


##############################################################################
"""functions to create SMOTE images"""

def biased_get_class(c):

    xbeg = [img for img, label in dataset if label == c]
    ybeg = [label for _, label in dataset if label == c]
    
    return xbeg, ybeg


def G_SM(X, y, n_to_sample,cl):

    # determining the number of samples to generate
    #n_to_sample = 10 

    # fitting the model
    n_neigh = 5 + 1
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

#xsamp, ysamp = SM(xclass,yclass)

###############################################################################


#NOTE: Download the training ('.../0_trn_img.txt') and label files 
# ('.../0_trn_lab.txt').  Place the files in directories (e.g., ../MNIST/trn_img/
# and /MNIST/trn_lab/).  Originally, when the code was written, it was for 5 fold
#cross validation and hence there were 5 files in each of the 
#directories.  Here, for illustration, we use only 1 training and 1 label
#file (e.g., '.../0_trn_img.txt' and '.../0_trn_lab.txt').

# dtrnimg = '.../MNIST/trn_img/'
# dtrnlab = '.../MNIST/trn_lab/'


# Define the path to your dataset
data_path = './smote/mod3/'

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the images to a fixed size
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

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
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

path_enc = "./smoke/models/crs5/0/bst_enc.pth"
path_dec = "./smoke/models/crs5/0/bst_dec.pth"

for i in range(1):
# for i in range(len(ids)):
    print()
    print(i)
    encoder = Encoder(args)
    decoder = Decoder(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    train_on_gpu = torch.cuda.is_available()

    #decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    
    batch_size = 100
    num_workers = 4

    best_loss = np.inf

    t0 = time.time()
    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])
    
        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0
            # train for one epoch -- set nets to train mode
            encoder.train()
            decoder.train()
        
            for j, (images,labs) in enumerate(train_loader):
            
                # zero gradients for each batch
                encoder.zero_grad()
                decoder.zero_grad()
                images, labs = images.to(device), labs.to(device)
                labsn = labs.detach().cpu().numpy()
            
                # run images
                z_hat = encoder(images)
            
                x_hat = decoder(z_hat) #decoder outputs tanh
                mse = criterion(x_hat,images)
                            
                resx = []
                resy = []
            
                tc = np.random.choice(len(dataset.class_to_idx.keys()),1)

                xbeg = [img.numpy() for img, label in dataset if label == tc]
                ybeg = [label for img, label in dataset if label == tc]
                xbeg = np.asarray(xbeg, dtype=float)
                ybeg = np.asarray(ybeg)

                xlen = len(xbeg)
                nsamp = min(xlen, 100)
                ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)
                xclass = xbeg[ind]
                yclass = ybeg[ind]
            
                xclen = len(xclass)
                xcminus = np.arange(1,xclen)
                
                xcplus = np.append(xcminus,0)
                xcnew = (xclass[[xcplus],:])
                xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
            
                xcnew = torch.Tensor(xcnew)
                xcnew = xcnew.to(device)
            
                #encode xclass to feature space
                xclass = torch.Tensor(xclass)
                xclass = xclass.to(device)
                xclass = encoder(xclass)
            
                xclass = xclass.detach().cpu().numpy()
            
                xc_enc = (xclass[[xcplus],:])
                xc_enc = np.squeeze(xc_enc)
            
                xc_enc = torch.Tensor(xc_enc)
                xc_enc = xc_enc.to(device)
                
                ximg = decoder(xc_enc)
                
                mse2 = criterion(ximg,xcnew)
            
                comb_loss = mse2 + mse
                comb_loss.backward()
            
                enc_optim.step()
                dec_optim.step()

                print("comb_loss", j, comb_loss.item())
            
                train_loss += comb_loss.item()*images.size(0)
                tmse_loss += mse.item()*images.size(0)
                tdiscr_loss += mse2.item()*images.size(0)
            
                 
            # print avg training statistics 
            train_loss = train_loss/len(train_loader)
            tmse_loss = tmse_loss/len(train_loader)
            tdiscr_loss = tdiscr_loss/len(train_loader)
            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                    train_loss,tmse_loss,tdiscr_loss))
            
        
        
            #store the best encoder and decoder models
            #here, /crs5 is a reference to 5 way cross validation, but is not
            #necessary for illustration purposes
            if train_loss < best_loss:
                print('Saving..')
                path_enc = './smoke/models/crs5/' \
                    + str(i) + '/bst_enc.pth'
                path_dec = './smoke/models/crs5/' \
                    + str(i) + '/bst_dec.pth'
             
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
        
                best_loss = train_loss
        
        
        #in addition, store the final model (may not be the best) for
        #informational purposes
        path_enc = './smoke/models/crs5/' \
            + str(i) + '/f_enc.pth'
        path_dec = './smoke/models/crs5/' \
            + str(i) + '/f_dec.pth'
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
        print()
              
    t1 = time.time()
    print('total time(min): {:.2f}'.format((t1 - t0)/60))             
 
t4 = time.time()
print('final time(min): {:.2f}'.format((t4 - t3)/60))

