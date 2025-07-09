import sys, os
sys.dont_write_bytecode = True
import pathlib
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config as cf

class ImageFolder_2dom(Dataset):
    IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

    def __init__(self, imgs_dir0, imgs_dir1): # 画像ファイルのパス一覧
        self.img_paths0 = self._get_img_paths(imgs_dir0)
        self.img_paths1 = self._get_img_paths(imgs_dir1)
        self.trans_crop = data_transforms_crop
        self.trans_only = data_transforms_only

    def __getitem__(self, idx):
        path0 = self.img_paths0[idx % len(self.img_paths0)]
        img0 = Image.open(path0).convert("RGB") # 画像読み込み
        w, h = img0.size
        if w == cf.cellSize and h == cf.cellSize: img0 = self.trans_only(img0)
        else: img0 = self.trans_crop(img0)

        path1 = self.img_paths1[idx % len(self.img_paths1)]
        img1 = Image.open(path1).convert("RGB") # 画像読み込み
        w, h = img1.size
        if w == cf.cellSize and h == cf.cellSize: img1 = self.trans_only(img1)
        else: img1 = self.trans_crop(img1)

        return img0, img1

    def _get_img_paths(self, img_dir): # 指定ディレクトリ内の画像ファイルパス一覧
        img_dir = pathlib.Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix in ImageFolder_2dom.IMG_EXT]
        return img_paths

    def __len__(self): # ディレクトリ内の画像ファイルの数
        dataset_len = len(self.img_paths0) if len(self.img_paths1) < len(self.img_paths0) else len(self.img_paths1) # (多い方の数にする)
        # dataset_len = len(self.img_paths0) if len(self.img_paths1) > len(self.img_paths0) else len(self.img_paths1) # (少ない方の数にする)
        return dataset_len * cf.dataAugRate

# データ変換
data_transforms_crop = T.Compose([
    # T.Resize(int(cf.cellSize * 1.2)),
    T.RandomRotation(degrees = 2, expand = True),
    # T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    # T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    T.RandomHorizontalFlip(0.5),
    # T.CenterCrop(cf.cellSize),
    T.RandomCrop((cf.cellSize, cf.cellSize), pad_if_needed = True),
    # T.TenCrop(cf.cellSize),
    T.ToTensor()
    ])
data_transforms_only = T.Compose([
    T.ToTensor()
    ])

def load_datasets(imgs_dir0, imgs_dir1):
    datasets_raw = ImageFolder_2dom(imgs_dir0, imgs_dir1)
    cf.dataset_size = len(datasets_raw)
    # print(cf.dataset_size)
    train_loader = DataLoader(datasets_raw, batch_size=cf.batchSize, shuffle=True, num_workers = 0, pin_memory=True)
    return train_loader

if __name__ == "__main__":
    import time
    f_tm = time.time()
    # python load_dataset.py /work/dataset/mrst/c_others/ /work/dataset/mrst/loundraw/

    dataloader = load_datasets(sys.argv[1], sys.argv[2])
    print(len(dataloader))
    if not os.path.exists("chk"): os.mkdir("chk")

    for n, (img0, img1) in enumerate(dataloader):
        # print(labels[n], imgpaths[n])
        print(n)
        print(img0.shape)
        print(img1.shape)
        buf_save_imgs = torch.cat([img0, img1], dim=0)
        torchvision.utils.save_image(buf_save_imgs, f"chk/_e_{n:05}.png", range=(-1.0,1.0), normalize=True)
        if n == 20: break

    print()
    print(time.time() - f_tm)
