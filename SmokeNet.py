#------------------------------------------ Documentation ---------------------------------
#   Author : Kshitij Bhat, Mihir Gadgil
#------------------------------------------------------------------------------------------

# %% ------------------------------------ Importing Libraries -----------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv

# %% ------------------------------------ Define Class -----------------------------

# W = np.random.rand((x_dim,y_dim))*np.sqrt(1/(ni+no))  -- Xavier initialization
class SmokeNet(nn.Module):
    def __init__(self, learn_rate = 0.001, ):
        # Call weight and bias initializer
        # initialize learning rate


        
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
        self.conv12= nn.Conv2d(512,512, kernel_size = (3,3), padding=1)
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

    def fit(self,)
        learn_rate = 0.001
        batch_size = 32
        weight_decay = 0.0001
        DROPOUT = 0.05
        N_EPOCHS =20
        for epoch in range(N_EPOCHS):

        loss_train = 0
        model.train()
        # m = nn.Sigmoid()
        min_loss = 2

        for batch in range(len(x_train)//BATCH_SIZE + 1):
            inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
            optimizer.zero_grad()
            logits = model(x_train[inds])

            loss = criterion(logits,y_train[inds].long())

            loss.backward()
            optimizer.step()
            loss_train += loss.item()


            # Check for validation loss instead of training loss

            
            if(min_loss > (loss_train/BATCH_SIZE)):
                print("=> Saving a new best")
                device = torch.device("cpu")
                model.to(device)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learn_rate': LR,
                    "DROPOUT": DROPOUT
                }, "model_smoke_detect.pt")
                

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)

    

    