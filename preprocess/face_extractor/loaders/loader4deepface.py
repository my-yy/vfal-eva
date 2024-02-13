import torch
import os
from deepface import DeepFace
import torchvision
from torchvision import transforms
import cv2
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, root_path, mode=None):
        all_image_files = []
        for person_name in os.listdir(root_path):
            for img in os.listdir(os.path.join(root_path, person_name)):
                if ".jpg" not in img:
                    continue
                all_image_files.append(os.path.join(root_path, person_name, img))
        self.all_image_files = all_image_files
        self.mode = mode

    def __len__(self):
        return len(self.all_image_files)

    def __getitem__(self, index):
        file_path = self.all_image_files[index]
        if self.mode == "facenet":
            img_PIL = Image.open(file_path)
            resize_size = 128
            transform_fn = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(resize_size, resize_size)),
                torchvision.transforms.ToTensor()
            ])
            data = transform_fn(img_PIL)

        elif self.mode == "deepface":
            resize_size = 224
            img = DeepFace.functions.preprocess_face(img=file_path, target_size=(resize_size, resize_size),
                                                     enforce_detection=False)
            img = img.squeeze(axis=0)
            data = torch.FloatTensor(img)
        else:
            data = load(file_path, 128)
        return data, index


def load(image_path, image_size):
    return trans_frame(cv2.imread(image_path), image_size)


def trans_frame(frame_npy, image_size):
    frame_pil = Image.fromarray(frame_npy)

    trans = torchvision.transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
    ])

    image = trans(frame_pil)
    return image
