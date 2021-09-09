import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class MLCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        #print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.load(img_path)
        input_image = image[0:4,:,:,:]
        target_image = image[4,:,:,:]
        
        return input_image, target_image

## TESTING SECTION ##    

if __name__ == "__main__":
    dataset = MLCDataset(config.TRAIN_DIR)
    loader = DataLoader(dataset, batch_size = 5)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        import sys
    
        sys.exit()