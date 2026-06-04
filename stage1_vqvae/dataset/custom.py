import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset

#from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

hw_ratio = 4

class ImagePaths(Dataset):
    def __init__(self, paths, size=256, random_crop=False, labels=None):
        self.size_h = size
        self.size_w = size * hw_ratio
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.padder = albumentations.PadIfNeeded(min_height=self.size_h, min_width=self.size_w)


    def __len__(self):
        return self._length
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        width, height = img.size
        ori_size = (width, height)
        img = np.array(img).astype(np.uint8)
        if height < self.size_h or height > self.size_h:
            new_width = int(self.size_h * width / height)
            resize = albumentations.Resize(height=self.size_h, width=new_width)
            img = resize(image = img)["image"]
            height , width, _ = img.shape
        if width > self.size_w:
            new_height = int(self.size_w * height / width)
            resize = albumentations.Resize(height=new_height, width=self.size_w)
            img = resize(image=img)["image"]
            height , width, _ = img.shape
        img = self.padder(image=img)["image"]
        img = (img / 127.5 - 1.0).astype(np.float32)
        return img

    def __getitem__(self, i):
        example = dict()
        try:
            example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        except:
            print("Error loading image from path: ", self.labels["file_path_"][i])
            return self.__getitem__(i+1)
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example




class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


