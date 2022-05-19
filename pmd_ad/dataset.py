import os
from PIL import Image
from torch.utils.data import Dataset

class PmdDataset(Dataset):

    def __init__(self, image_dir, mask_dir, edge_dir, transform_data, transform_label):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir

        self.transform_image = transform_data
        self.transform_mask = transform_label
        self.transform_edge = transform_label

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg','.png'))
        edge_path = os.path.join(self.edge_dir, self.images[index].replace('.jpg','.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")
        edge = Image.open(edge_path).convert("1")

        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        edge = self.transform_edge(edge)

        return image, mask, edge

