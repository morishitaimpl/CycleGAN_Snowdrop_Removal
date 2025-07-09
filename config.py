import sys, random
sys.dont_write_bytecode = True
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 画像の一辺のサイズ (この大きさにリサイズされるので要確認)
cellSize = 512

# 繰り返す回数
# epochSize = 100
epochSize = 3

# 学習するときの小さいセットの数：GANなので少なめも要確認
batchSize = 4

# データセットの数 (イテレーション数を求めたりするためにグローバルで使えるようにしておく)
dataset_size = 0

# 割り増しする倍率
dataAugRate = 1

# モデル内のresnetの深さ
if cellSize == 128: resBlocks=6
else: resBlocks=9

# identity損失：0〜1
# lambda_identity = 0
lambda_identity = 1

# cycle損失：0〜10
# lambda_cycle = 10
lambda_cycle = 1

class loss_scheduler():
    def __init__(self, epoch_decay):
        self.epoch_decay = epoch_decay

    def f(self, epoch):
        #ベースの学習率に対する倍率を返す(pytorch仕様)
        if epoch<=self.epoch_decay:
            return 1
        else:
            scaling = 1 - (epoch-self.epoch_decay)/float(self.epoch_decay)
            return scaling

def set_requires_grad(models, requires=False):
    if not isinstance(models,list):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires

class ImagePool():
    def __init__(self,pool_size=50):
        self.pool_size = pool_size
        self.buffer = []
    
    def get_images(self,pre_images):
        return_imgs = []
        for img in pre_images:
            # pdb.set_trace()
            img = torch.unsqueeze(img,0)
            if len(self.buffer) < self.pool_size:
                self.buffer.append(img)
                return_imgs.append(img)
            else:
                if random.randint(0,1)>0.5:
                    i = random.randint(0,self.pool_size-1)
                    tmp = self.buffer[i].clone()
                    self.buffer[i]=img
                    return_imgs.append(tmp)
                else:
                    return_imgs.append(img)
        return torch.cat(return_imgs,dim=0)

def init_weights(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(net.weight.data, 0.0, 0.02)
        if hasattr(net, 'bias') and net.bias is not None:
            torch.nn.init.constant_(net.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(cellSize,cellSize,kernel_size=3,stride=1),
            nn.InstanceNorm2d(cellSize),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(cellSize,cellSize,kernel_size=3,stride=1),
            nn.InstanceNorm2d(cellSize)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x

class Generator(nn.Module):
    def __init__(self, img_channel, res_block):
        super().__init__()
        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channel,64,kernel_size=7,stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,512,kernel_size=3,stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResidualBlock() for _ in range(res_block)]
        self.res_block = nn.Sequential(
            *res_blocks
        )
        self.decode_block = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64,img_channel,kernel_size=7,stride=1),
            nn.Tanh()
        )

    
    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        x = self.decode_block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,img_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(img_channel,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_gen = Generator(3, 6).to(DEVICE)
    print(mdl_gen)
    summary(mdl_gen, (3, cellSize, cellSize))

    mdl_dis = Discriminator(3).to(DEVICE)
    print(mdl_dis)
    summary(mdl_dis, (3, cellSize, cellSize))