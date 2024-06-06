from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import mrcfile


class TomographyDataset(Dataset):
    # Initialization function
    def __init__(
            self,
            data_folder_path: str,
            max_files: int=None,
            additional_transforms: list=[]
        ):
        super().__init__()

        # Set the max files variable. If it's none, then have it be 
        self.max_files = max_files if max_files is not None else float('inf')

        # Have it search for all of the .mrc files
        self.fpaths = []
        path, root_folders, _ = next(os.walk(data_folder_path))
        root_folders.remove("fslg_documents")

        for folder in root_folders:
            # Recursively search through all of the data directories for the .mrc files
            self.__data_folder_path_helper(path, folder)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            *additional_transforms
        ])

    def __data_folder_path_helper(self, fpath, top_folder):
        cwd = os.path.join(fpath, top_folder)
        path, folders, files = next(os.walk(cwd))

        for file in files:
            if ".rec" in file:
                self.fpaths.append(os.path.join(cwd, file))

        if len(self.fpaths) < self.max_files:
            for folder in folders:
                # Makes sure that the folder isn't a hidden folder.
                if folder[0] != ".":
                    self.__data_folder_path_helper(path, folder)

    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, index):
        fpath = self.fpaths[index]
        mrc_file = mrcfile.read(fpath)

        # Iterates through the transformations
        return self.transforms(mrc_file)
    
    
    
if __name__ == "__main__":
    dataset = TomographyDataset(
        data_folder_path="/home/mwh1998/fsl_groups/",
        max_files=100,
        additional_transforms=[]
    )
    