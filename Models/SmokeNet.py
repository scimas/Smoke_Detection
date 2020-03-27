#------------------------------------------ Documentation ---------------------------------
#   Author : Kshitij Bhat, Mihir Gadgil
#------------------------------------------------------------------------------------------

# %% ------------------------------------ Importing Libraries -----------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
from SmokeDataset import get_datasets
from torch.utils.data import DataLoader

# %% ------------------------------------ Define Class -----------------------------

# W = np.random.rand((x_dim,y_dim))*np.sqrt(1/(ni+no))  -- Xavier initialization
class SmokeNet(nn.Module):
    def __init__(self, learn_rate = 0.001, n_epochs=20, batch_size = 50, dropout = 0.5):
        # Call weight and bias initializer
        # initialize learning rate



        self.learn_rate = learn_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss()
     
        super(SmokeNet, self).__init__()
        # set the channels and dimensions later
        # Initial size of the array 3 x 224 X 224
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride = (2,2), padding=3) # 112 x 112 x 32
        # self.convnorm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)) # 56 x 56 x 64

        # First block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1,1)) 
        self.conv3 = nn.Conv2d(64,256, kernel_size=(3,3), padding= 1)
        self.conv4 = nn.Conv2d(256,128, kernel_size=(1,1)) # 128 X 56 X 56

        self.pool2 = nn.AvgPool2d(kernel_size=(1,1))

        #self.ra1
        # global average pooling
        # Dense blocks for RA-SC here 

        # Second block
        self.conv5 = nn.Conv2d(128,128, kernel_size=(1,1),stride = (2,2))
        self.conv6 = nn.Conv2d(128,128, kernel_size=(3,3), padding= 1)
        self.conv7 = nn.Conv2d(128,512, kernel_size=(1,1)) # 512 X 28 X 28

        self.pool3 = nn.AvgPool2d(kernel_size=(1,1))

        # self.ra2
        # global average pooling
        # Dense block for RA-SC



        # Third block 
        self.conv8 = nn.Conv2d(128,256 , kernel_size=(1,1), stride=(2,2))
        self.conv9 = nn.Conv2d(256,256, kernel_size = (3,3), padding=1)
        self.conv10 = nn.Conv2d(256,1024, kernel_size=(1,1))

        self.pool4 = nn.AvgPool2d(kernel_size=(1,1))

        # self.ra3
        # global average pooling
        # Dense block for RA-SC



        # Fourth Block

        self.conv11 = nn.Conv2d(1024,512 , kernel_size=(1,1), stride=(2,2))
        self.conv12 = nn.Conv2d(512,512, kernel_size = (3,3), padding=1)
        self.conv13 = nn.Conv2d(512,2048, kernel_size=(1,1))

        self.pool5 = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))

        # self.avgpool

        ## Final average pool 7 x 7, stride 1


        self.layers = nn.ModuleList( [self.conv1, self.pool1 ,self.conv2,self.conv3, self.conv4,self.ra1,self.conv5, self.conv6, 
        self.conv7, self.ra2, self.conv8, self.conv9, self.conv10, self.ra3,  self.conv11, self.conv12, self.conv13 , self.avgpool])

        # fc

        # Apply softmax for  ROC in testing

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def fit(self, train_data , validation_data):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        #initialize train and validation data loaders
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
        validation_loader = DataLoader(validation_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

        print("######The device is: ", device)
        print("Starting training loop...")
        for epoch in range(self.n_epochs):
            loss_train = 0
            self.train()
            min_loss = 2
            for i, (images, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                logits = self.forward(images.to(device))
                loss = self.criterion(logits,labels.to(device))
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item() * len(labels)

                # Average training loss. Less meaningful since model is being updated on each minibatch.
                loss_train /= len(train_loader)
                with torch.no_grad():
                    validation_loss = 0
                    self.eval()
                    for i, (images, labels) in enumerate(validation_loader):
                        logits = self.forward(images.to(device))
                        validation_loss += self.criterion(logits,labels.to(device)).item() * len(labels)
                        # Average validation loss.
                        validation_loss /= len(validation_loader)
                        if validation_loss < min_validation_loss:
                            min_validation_loss = validation_loss
                            print("=> Saving a new best")
                            torch.save({
                                'model_state_dict': self.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()}, "model_smokenet.pt")
                        else:
                            print("=> Validation loss did not improve")
                            print("Epoch {} | Validation Loss {:.5f}".format(epoch, validation_loss))

    
def predict(test_data):
    # initialize test data loader

    # test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    #determine if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #initialize model
    model = SmokeNet()

    # load the saved model
    checkpoint = torch.load("model_smokenet.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.SGD(model.parameters(), lr=checkpoint['learn_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    x_test = test_data[1].to(device)

    with torch.no_grad():
        logits = model(x_test)
        print(logits)
        y_pred = torch.from_numpy((logits.cpu()>0.5).numpy()).float()
        print(y_pred.shape)
        print(y_pred)
    return y_pred
