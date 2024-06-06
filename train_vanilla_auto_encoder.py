import torch
from data import (
    TomographyDataset,
    RandomRotation3D,
    RandomSlice3D,
    CropTomography,
    CenterCrop3D
)
from models import Conv3DAutoEncoder
import lightning as L
from pytorch_lightning import loggers
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import argparse

def full_run():
    # Hyperparameters
    chunk_size = 64
    number_of_files = 100
    lr=1e-3
    batch_size=10
    
    # Set up dataset related 
    dataset = TomographyDataset(
        "/home/mwh1998/fsl_groups/",
        chunk_size,
        number_of_files
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )
    
    model = Conv3DAutoEncoder(
        torch.tensor(chunk_size)
    )
    
    logger = loggers.TensorBoardLogger("./logs")
    trainer = L.Trainer(
        accelerator="gpu",
        logger=logger
    )

    trainer.fit(model, data_loader)

def test_code():
    chunk_size = 64
    
    dataset = TomographyDataset(
        "/home/mwh1998/fsl_groups",
        max_files=100
    )

    transforms = Compose([
            CropTomography(32),
            RandomSlice3D( 
                torch.sqrt(3 * torch.pow(torch.tensor(chunk_size),2))
            ),
            RandomRotation3D(
                -15, 15,
                0, 360,
                -15, 15
            ),
            CenterCrop3D(64)
        ]
    )

    sample = dataset[0]

    transformed_sample = transforms(sample)

    print(transformed_sample.size())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    if args.test:
        test_code()
    else:
        full_run()