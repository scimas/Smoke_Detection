#------------------------------------------ Documentation ---------------------------------
#   Author : Kshitij Bhat, Mihir Gadgil
#------------------------------------------------------------------------------------------

# %% ------------------------------------ Importing Libraries -----------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau



class SmokeNet(nn.Module):
    def __init__(self, sc_cs="SC"):
        # Call weight and bias initializer
        # initialize learning rate
        self.red_ratio = 16

        super(SmokeNet, self).__init__()
        # Initial size of the array 3 x 224 X 224
        top_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)) # 64 x 112 x 112
        top_act = nn.ReLU()
        top_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # 64 x 56 x 56

        # First block
        block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=(1, 1)), # 256 X 56 X 56
            nn.ReLU()
        )

        ra1 = ResidualAttention(channels=256, height=56, width=56, n=2, red_ratio=self.red_ratio, variant=sc_cs)

        # Second block
        block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=(1, 1)), # 512 X 28 X 28
            nn.ReLU()
        )

        ra2 = ResidualAttention(channels=512, height=28, width=28, n=1, red_ratio=self.red_ratio, variant=sc_cs)

        # Third block
        block3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=(1, 1)), # 1024 x 14 x 14
            nn.ReLU()
        )

        ra3 = ResidualAttention(channels=1024, height=14, width=14, n=0, red_ratio=self.red_ratio, variant=sc_cs)

        # Fourth Block
        block4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=(1, 1)), # 2048 x 7 x 7
            nn.ReLU()
        )

        # Final average pool 7 x 7, stride 1
        last_pool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1)) # 2048 x 1 x 1

        # Final prediction layer, flatten and linear
        flat = nn.Flatten()
        fc = nn.Linear(2048, 6)

        self.layers = nn.Sequential(
            top_conv, top_act, top_pool,
            block1, ra1,
            block2, ra2,
            block3, ra3,
            block4, last_pool,
            flat, fc
        )

    def forward(self, x):
        return self.layers(x)


def fit(model, optimizer, criterion, train_data, validation_data, class_weights=None, n_epochs=100, batch_size=32, filename="model"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize train and validation data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # Reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    logfile = open(filename+".log","w")
    log_df = pd.DataFrame({"Epoch":np.arange(n_epochs), "Training_Loss": np.zeros(n_epochs), "Validation_Loss":np.zeros(n_epochs)})



    min_validation_loss = 1e10
    print("######The device is: ", device)
    print("Starting training loop...")
    for epoch in range(n_epochs):
        loss_train = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_train += loss.item() * len(labels)
        # Average training loss. Less meaningful since model is being updated on each minibatch.
        loss_train /= len(train_loader)
        with torch.no_grad():
            validation_loss = 0
            model.eval()
            for i, (images, labels) in enumerate(validation_loader):
                logits = model(images.to(device))
                validation_loss += criterion(logits, labels.to(device)).item() * len(labels)
            # Average validation loss.
            validation_loss /= len(validation_loader)

            # reduce lr if the validation has stopped improving/decreasing
            scheduler.step(validation_loss)

            # save the model when the validation_loss has improved/decresaed
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                print("=> Saving a new best")
                if hasattr(model, "module"):
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, "model_smokenet.pt")
                else:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, "model_smokenet.pt")
            else:
                print("=> Validation loss did not improve")
            print("Epoch {} | Training loss {:.5f} | Validation Loss {:.5f}".format(epoch, loss_train, validation_loss))
            logfile.write("Epoch {} | Training loss {:.5f} | Validation Loss {:.5f} \n".format(epoch, loss_train, validation_loss))
            log_df.loc[epoch,"Training_Loss"] = loss_train
            log_df.loc[epoch,"Validation_Loss"] = validation_loss
    optimizer.zero_grad()
    logfile.close()
    log_df.to_csv(filename+".csv", index=False)


def predict(model, test_data):
    # initialize test data loader
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    model.eval()
    # determine if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preds = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            logits = model(images.to(device))
            y_pred = nn.functional.softmax(logits, dim=1).cpu()
            preds.extend(y_pred)
    return preds


class Spatial_Attention(nn.Module):
    def __init__(self, H=56, W=56, in_channels=128, red_ratio=16):
        self.in_channels = in_channels
        self.H = H
        self.W = W
        self.out_channels = self.in_channels // red_ratio
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, 1)
        self.conv2 = nn.Conv1d(self.out_channels, 1, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.H * self.W)
        s_attn_dist = self.sig(self.conv2(self.relu(self.conv1(x))))
        x = torch.reshape(x * s_attn_dist, (-1, self.in_channels, self.H, self.W))
        return x


class Channel_Attention(nn.Module):
    def __init__(self, H=56, W=56, in_channels=128, red_ratio=16):
        self.in_channels = in_channels
        self.H = H
        self.W = W
        self.out_channels = self.in_channels // red_ratio
        super(Channel_Attention, self).__init__()

        self.gavg = nn.AvgPool1d(H * W)
        self.lin1 = nn.Linear(self.in_channels, self.out_channels)
        self.lin2 = nn.Linear(self.out_channels, self.in_channels)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.H * self.W)
        c_att = self.sig(
            self.lin2(
                self.relu(
                    self.lin1(
                        self.gavg(x).reshape(-1, self.in_channels)
                    )
                )
            )
        ).reshape(-1, self.in_channels, 1)
        x = torch.reshape(x * c_att, (-1, self.in_channels, self.H, self.W))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super(ResidualBlock, self).__init__()
        if downsample:
            conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1))
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride=(2, 2)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1))
            if in_channels != out_channels:
                shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1, 1)),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                shortcut = nn.Identity()
        self.shortcut = shortcut
        act1 = nn.ReLU()
        norm1 = nn.BatchNorm2d(out_channels)
        normact = nn.ReLU()
        conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1))
        act2 = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU()
        self.layers = nn.Sequential(
            conv1, act1, norm1, normact, conv2, act2
        )

    def forward(self, x):
        out1 = self.shortcut(x)
        out2 = self.layers(x)
        return self.act3(self.norm(out1 + out2))


def make_RA_block(channels: int, height: int, width: int, red_ratio: int, variant: str):
    variant = variant.lower()
    if variant == "sc":
        return nn.Sequential(
            ResidualBlock(channels, channels, False),
            Spatial_Attention(height, width, channels, red_ratio),
            Channel_Attention(height, width, channels, red_ratio)
        )
    elif variant == "cs":
        return nn.Sequential(
            ResidualBlock(channels, channels, False),
            Channel_Attention(height, width, channels, red_ratio),
            Spatial_Attention(height, width, channels, red_ratio)
        )
    elif variant == "c":
        return nn.Sequential(
            ResidualBlock(channels, channels, False),
            Channel_Attention(height, width, channels, red_ratio)
        )
    elif variant == "s":
        return nn.Sequential(
            ResidualBlock(channels, channels, False),
            Spatial_Attention(height, width, channels, red_ratio)
        )
    else:
        raise ValueError("invalid RA block variant, can only be 'sc' or 'cs'")


class ResidualAttention(nn.Module):
    def __init__(self, channels: int, height: int, width: int, n: int, red_ratio: int, variant: str):
        super(ResidualAttention, self).__init__()
        # Top block
        self.top_block = make_RA_block(channels, height, width, red_ratio, variant)
        # Trunk branch
        self.trunck_branch = nn.Sequential(
            make_RA_block(channels, height, width, red_ratio, variant),
            make_RA_block(channels, height, width, red_ratio, variant)
        )
        # Soft mask branch
        self.soft_mask_branch = nn.ModuleDict()
        # First pooling layer -> downsample to half size
        self.soft_mask_branch["pool1"] = nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1))
        height, width = height // 2, width // 2
        # First RA block after first downsample
        self.soft_mask_branch["ra1"] = make_RA_block(channels, height, width, red_ratio, variant)
        # The side RA block in soft mask branch
        self.soft_mask_branch["side_ra"] = make_RA_block(channels, height, width, red_ratio, variant)
        # `n` downsamplers
        downsamplers = nn.ModuleList()
        for i in range(n):
            downsamplers.append(nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)))
            height, width = height // 2, width // 2
            downsamplers.append(make_RA_block(channels, height, width, red_ratio, variant))
        # Sometimes there is an extra RA block before upsamplers.
        # Add a dummy identity if the RA block isn't needed: avoid an if check in forward
        if n > 1:
            ra_mid = make_RA_block(channels, height, width, red_ratio, variant)
        else:
            ra_mid = nn.Identity()
        # `n` upsamplers
        upsamplers = nn.ModuleList()
        for i in range(n):
            upsamplers.append(make_RA_block(channels, height, width, red_ratio, variant))
            upsamplers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            height, width = height * 2, width * 2
        # Convert the downsamplers, ra_mid and upsamplers into an nn.Sequential
        # Easier to execute in forward
        self.soft_mask_branch["down-up"] = nn.Sequential(*downsamplers, ra_mid, *upsamplers)
        # The bottom RA block
        self.soft_mask_branch["ra2"] = make_RA_block(channels, height, width, red_ratio, variant)
        # Final bilinear upsampling interpolation that undoes 'pool1' downsampling
        self.soft_mask_branch["upsample"] = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # Final 1x1 convolutions and sigmoid
        self.soft_mask_branch["convs"] = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(channels, channels, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.top_block(x)
        out_trunk = self.trunck_branch(x)
        # Soft mask calculations
        out1 = self.soft_mask_branch["ra1"](
            self.soft_mask_branch["pool1"](x)
        )
        out_side = self.soft_mask_branch["side_ra"](out1)
        down_up = self.soft_mask_branch["down-up"](out1)
        out_soft = self.soft_mask_branch["ra2"](out_side + down_up)
        out_soft = self.soft_mask_branch["upsample"](out_soft)
        out_soft = self.soft_mask_branch["convs"](out_soft)
        return out_trunk * (1 + out_soft)
