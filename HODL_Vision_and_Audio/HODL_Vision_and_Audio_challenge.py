import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # skip connection if input/output channel different
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv(x)
        return self.activation(out + residual)


class ImprovedUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = ResidualConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = ResidualConvBlock(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.down3 = ResidualConvBlock(128, 256)
        self.pool3 = nn.MaxPool1d(2)
        self.down4 = ResidualConvBlock(256, 512)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ResidualConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(512, 256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(256, 128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(128, 64)

        # output layer
        self.final = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=1),
            nn.Tanh()  # constrain to [-1, 1]
        )

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(b)
        u4 = self._crop_and_concat(c4, u4)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = self._crop_and_concat(c3, u3)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = self._crop_and_concat(c2, u2)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = self._crop_and_concat(c1, u1)
        d1 = self.dec1(u1)

        return self.final(d1)

    def _crop_and_concat(self, enc_feat, dec_feat):
        diff = enc_feat.size(-1) - dec_feat.size(-1)
        if diff > 0:
            enc_feat = enc_feat[:, :, :-diff]
        return torch.cat([enc_feat, dec_feat], dim=1)


# ----------- Initialization Function ------------
def init_model() -> nn.Module:
    model = ImprovedUNet1D(in_channels=1, out_channels=1)
    return model


# ----------- Training Function ------------
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.L1Loss()  

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

    return model

