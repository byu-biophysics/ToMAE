import torch.nn as nn
import torch.nn.functional as f
import torch
import lightning as L

class ResidualBlock3D(nn.Module):
    def __init__(self, i_c: int, o_c: int):
        super().__init__()

        self.conv1 = nn.Conv3d(
            i_c,
            o_c,
            kernel_size=(1,1,1)
        )

        self.conv2 = nn.Conv3d(
            in_channels=o_c,
            out_channels=o_c,
            kernel_size=(3,3,3),
            padding=(1,1,1)
        )

        self.bn1 = nn.BatchNorm3d(
            num_features=o_c
        )

        self.conv3 = nn.Conv3d(
            in_channels=o_c,
            out_channels=o_c,
            kernel_size=(3,3,3),
            padding=(1,1,1)
        )

        self.bn2 = nn.BatchNorm3d(
            num_features=o_c
        )

        self.conv4 = nn.Conv3d(
            in_channels=o_c,
            out_channels=i_c,
            kernel_size=(1,1,1)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = f.silu(self.bn1(self.conv2(x)))
        x = f.silu(self.bn2(self.conv3(x)))
        return self.conv4(x)

class DownConv3DBlock(nn.Module):
    def __init__(
            self,
            i_c: int,
            o_c: int,
            num_res_blocks: int = 2,
            dropout_prob: str = 0.5
        ):
        super().__init__()
        
        self.initial_conv = nn.Conv3d(i_c, o_c, kernel_size=(3,3,3), padding=(1,1,1))

        self.residual_blocks = nn.ModuleList([ResidualBlock3D(o_c, o_c) for i in range(num_res_blocks)])

        self.pool = nn.MaxPool3d(
            kernel_size=(2,2,2)
        )

        self.dropout = nn.Dropout3d(
            p=dropout_prob
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x) + x
        
        return self.dropout(self.pool(x))

class Conv3DEncoder(nn.Module):
    def __init__(
            self,
            num_blocks: int,
            channel_power: int = 2, # Each layer will subsequently have channel_power^n channels. So if it's the first layer, it will have 4 channels as the output
            dropout_prob: float = 0.5,
            embedding_size: int = 128
        ):
        super().__init__()

        self.initial_conv = nn.Conv3d(1, channel_power, kernel_size=(3,3,3), padding=(1,1,1))
        self.initial_do = nn.Dropout3d(p=dropout_prob)
        self.initial_bn = nn.BatchNorm3d(channel_power)

        modules = []
        in_channels = channel_power
        for i in range(num_blocks):
            out_channels = (channel_power + 2) ** (i+1)
            modules.append(DownConv3DBlock(in_channels, out_channels, dropout_prob=0.5))
            in_channels = out_channels
        
        self.downblocks = nn.ModuleList(modules)

        linear_channel_count = (channel_power + 2) ** (num_blocks)

        compression_layers = []
        while linear_channel_count != embedding_size * 2:
            output_count = linear_channel_count // 2
            compression_layers.append(nn.Linear(linear_channel_count, output_count))
            compression_layers.append(nn.Dropout1d(p=dropout_prob))
            compression_layers.append(nn.SiLU())
            linear_channel_count = linear_channel_count // 2

        self.compression_layers = nn.ModuleList(compression_layers)

        self.final_layer = nn.Linear(linear_channel_count, embedding_size)

    def forward(self, x) -> torch.Tensor:
        x = f.silu(self.initial_bn(self.initial_do(self.initial_conv(x))))

        for downblock in self.downblocks:
            x = downblock(x)
        
        x = x.squeeze()
        for layer in self.compression_layers:
            x = layer(x)

        return self.final_layer(x)

class UpConv3DBlock(nn.Module):
    def __init__(self,
        i_c: int,
        o_c: int,
        num_res_blocks: int = 2,
        dropout_prob: str = 0.5
    ):
        super().__init__()
        self.initial_conv = nn.Conv3d(i_c, i_c, kernel_size=(3,3,3), padding=(1,1,1))
        self.residual_blocks = nn.ModuleList([ResidualBlock3D(i_c, i_c) for i in range(num_res_blocks)])

        self.conv_upsample = nn.ConvTranspose3d(i_c, o_c, (2,2,2), stride=(2,2,2))
        self.dropout = nn.Dropout3d(p=dropout_prob)

    def forward(self, x) -> torch.Tensor:
        x = self.initial_conv(x)

        for block in self.residual_blocks:
            x = block(x) + x
        
        return self.dropout(f.relu(self.conv_upsample(x))) 

class Conv3DDecoder(nn.Module):
    def __init__(
            self,
            num_blocks: int,
            channel_power: int = 2, # Each layer will subsequently have channel_power^n channels. So if it's the first layer, it will have 4 channels as the output
            dropout_prob: float = 0.5,
            embedding_size: int = 128
        ):
        super().__init__()

        current_input_size = embedding_size
        current_output_size = 0

        self.final_linear_size = (channel_power + 2) ** (num_blocks)

        decompression_layers = []
        while current_output_size != self.final_linear_size:
            current_output_size = current_input_size * 2
            decompression_layers.append(nn.Linear(current_input_size, current_output_size))
            decompression_layers.append(nn.Dropout1d(p=dropout_prob))
            decompression_layers.append(nn.SiLU())
            current_input_size = current_output_size
        self.decompression_layers = nn.ModuleList(decompression_layers)

        upblocks = []
        current_input_size = self.final_linear_size
        for i in range(num_blocks, 0, -1):
            
            if i != 1:
                current_output_size = (channel_power + 2) ** (i-1)
            else:
                current_output_size = channel_power

            upblocks.append(UpConv3DBlock(current_input_size, current_output_size))
            current_input_size = current_output_size

        self.upblocks = nn.ModuleList(upblocks)

        self.final_conv = nn.Conv3d(channel_power, 1, kernel_size=(3,3,3), padding=(1,1,1))

    def forward(self, x) -> torch.Tensor:
        for layer in self.decompression_layers:
            x = layer(x)

        x = x.view(x.size()[0], self.final_linear_size, 1, 1, 1)
        
        for upblock in self.upblocks:
            x = upblock(x)

        return self.final_conv(x)

class Conv3DAutoEncoder(L.LightningModule):
    def __init__(
            self,
            chunk_size,
            dropout_prob: float = 0.5,
            embedding_size: int = 256
        ):
        super().__init__()
        
        num_blocks = int(torch.floor(torch.log2(chunk_size))) // 2

        self.encoder = Conv3DEncoder(
            num_blocks,
            dropout_prob=dropout_prob,
            embedding_size=embedding_size
        )
        
        self.decoder = Conv3DDecoder(
            num_blocks,
            dropout_prob=dropout_prob,
            embedding_size=embedding_size
        )

    def forward(self, x) -> torch.Tensor:
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters, lr=1e-3)
        return optimizer
    
    def training_step(self, x, batch_idx):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = f.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, x, batch_idx):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = f.mse_loss(x_hat, x)
        self.log("validation_step", loss, on_step=True, on_epoch=True)
        return loss

    
if __name__ == "__main__":
    cs = 64
    sim_chunk = torch.rand((10,1,cs,cs,cs))

    network = Conv3DAutoEncoder(torch.tensor(cs))
    print(sum(p.numel() for p in network.parameters() if p.requires_grad))

    with torch.no_grad():
        output = network(sim_chunk)
        print(output.size())
