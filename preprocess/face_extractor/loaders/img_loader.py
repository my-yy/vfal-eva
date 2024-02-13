import torch
from PIL import Image
import torchvision


class Dataset(torch.utils.data.Dataset):

    def __init__(self, all_image_files):
        self.all_image_files = all_image_files
        resize_size = 128
        self.transform_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(resize_size, resize_size)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.all_image_files)

    def __getitem__(self, index):
        file_path = self.all_image_files[index]
        img_PIL = Image.open(file_path)
        if img_PIL.mode != "RGB":
            img_PIL = img_PIL.convert("RGB")

        data = self.transform_fn(img_PIL)
        assert data.shape == (3, 128, 128), file_path
        return data, index
