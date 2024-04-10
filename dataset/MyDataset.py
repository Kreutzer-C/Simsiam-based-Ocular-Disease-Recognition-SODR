from torch.utils.data import Dataset
from PIL import Image
import os

class ODIRDataSet_Sim(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_list[item])
        img = Image.open(img_path).convert('RGB')

        x1, x2 = self.transform(img)

        return x1, x2

class ODIRDataSet_ft(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_path1 = os.path.join('../dataset/ODIR-5K_Training_Dataset', self.df.iloc[item, 0])
        img_path2 = os.path.join('../dataset/ODIR-5K_Training_Dataset', self.df.iloc[item, 1])
        labels = self.df.iloc[item, 2:].values.astype("float32")

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, labels

class ODIRDataSet_oltest(Dataset):
    def __init__(self, dir, transform=None):
        self.img_list = os.listdir(dir)
        self.id_list = self.get_id()
        self.transform = transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        id = self.id_list[item]
        img_path1 = os.path.join('../dataset/ODIR-5K_Testing_Images', id + "_left.jpg")
        img_path2 = os.path.join('../dataset/ODIR-5K_Testing_Images', id + "_right.jpg")

        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return id, img1, img2

    def get_id(self):
        id_list = []
        for filename in self.img_list:
            if filename.endswith("_left.jpg"):
                id = filename.replace("_left.jpg", "")
                id_list.append(id)
        return id_list
