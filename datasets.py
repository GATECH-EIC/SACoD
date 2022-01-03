import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', portion=None):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self._portion = portion

        self.files_A_total = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B_total = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

        if self._portion is not None:
            num_files_A = len(self.files_A_total)
            num_files_B = len(self.files_B_total)

            if self._portion > 0:
                split_A = int(np.floor(self._portion * num_files_A))
                self.files_A = self.files_A_total[:split_A]

                split_B = int(np.floor(self._portion * num_files_B))
                self.files_B = self.files_B_total[:split_B]   

            elif self._portion < 0:
                split_A = int(np.floor((1 + self._portion) * num_files_A))
                self.files_A = self.files_A_total[split_A:]

                split_B = int(np.floor((1 + self._portion) * num_files_B))
                self.files_B = self.files_B_total[split_B:]

        else:
            self.files_A = self.files_A_total
            self.files_B = self.files_B_total


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        # return max(len(self.files_A), len(self.files_B))
        return len(self.files_A)