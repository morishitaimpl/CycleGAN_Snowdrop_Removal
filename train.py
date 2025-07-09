import sys, time, os
sys.dont_write_bytecode = True
import numpy as np
import statistics

import torch
from torch import nn
import torchvision
from itertools import chain

import load_dataset as ld
import config as cf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
id_str = sys.argv[1]
id_str += f"_b{cf.batchSize}_i{cf.lambda_identity}_c{cf.lambda_cycle}"
dataset_path_src = sys.argv[2]
dataset_path_target = sys.argv[3]
path_log = "_l_" + id_str + ".csv"
log_dir = "_log_" + id_str
if not os.path.exists(log_dir): os.mkdir(log_dir) # 画像保存用のフォルダ

# モデルの定義
G_A2B = cf.Generator(3,cf.resBlocks).to(DEVICE)
G_B2A = cf.Generator(3,cf.resBlocks).to(DEVICE)
D_A = cf.Discriminator(3).to(DEVICE)
D_B = cf.Discriminator(3).to(DEVICE)

# 重みの初期化
G_A2B.apply(cf.init_weights)
G_B2A.apply(cf.init_weights)
D_A.apply(cf.init_weights)
D_B.apply(cf.init_weights)

model_G = G_A2B
model_D = G_B2A

# 各種の損失関数
adv_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()

# 各種の最適化関数
optimizer_G = torch.optim.Adam(chain(G_A2B.parameters(),G_B2A.parameters()),lr=0.0002,betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=cf.loss_scheduler(100).f)
scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,lr_lambda=cf.loss_scheduler(100).f)
scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,lr_lambda=cf.loss_scheduler(100).f)

fake_A_buffer = cf.ImagePool()
fake_B_buffer = cf.ImagePool()

# 学習
dataset = ld.load_datasets(dataset_path_src, dataset_path_target)
itr_size = cf.dataset_size // cf.batchSize
s_tm = time.time()
with open(path_log, mode = "w") as f: print("gab_mse,gba_mse,gca_l1,gcb_l1,da_mse,db_mse", file = f) # 損失推移の記録用
for i in range(cf.epochSize):
    ll_G_A2B, ll_G_B2A, ll_G_CA, ll_G_CB, ll_D_A, ll_D_B = [], [], [], [], [], []
    n_tm = time.time()
    losses = [0 for i in range(6)]
    for n, (imgs_src, img_target) in enumerate(dataset):
        batch_len = len(imgs_src)
        # 画像の準備
        real_A, real_B = imgs_src.to(DEVICE), img_target.to(DEVICE)
        fake_A, fake_B = G_B2A(real_B), G_A2B(real_A)
        rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)
        if cf.lambda_identity > 0: iden_A, iden_B = G_B2A(real_A), G_A2B(real_B)

        # generatorの学習
        cf.set_requires_grad([D_A, D_B],False)
        optimizer_G.zero_grad()

        pred_fake_A = D_A(fake_A)
        loss_G_B2A = adv_loss(pred_fake_A, torch.tensor(1.0).expand_as(pred_fake_A).to(DEVICE))

        pred_fake_B = D_B(fake_B)
        loss_G_A2B = adv_loss(pred_fake_B, torch.tensor(1.0).expand_as(pred_fake_B).to(DEVICE))

        loss_cycle_A = cycle_loss(rec_A, real_A)
        loss_cycle_B = cycle_loss(rec_B, real_B)

        if cf.lambda_identity > 0:
            loss_identity_A = identity_loss(iden_A, real_A)
            loss_identity_B = identity_loss(iden_B, real_B)
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A * cf.lambda_cycle + loss_cycle_B * cf.lambda_cycle + loss_identity_A * cf.lambda_cycle * cf.lambda_identity + loss_identity_B * cf.lambda_cycle * cf.lambda_identity
        else:
            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A * cf.lambda_cycle + loss_cycle_B * cf.lambda_cycle

        loss_G.backward()
        optimizer_G.step()

        ll_G_A2B.append(loss_G_A2B.item())
        ll_G_B2A.append(loss_G_B2A.item())
        ll_G_CA.append(loss_cycle_A.item())
        ll_G_CB.append(loss_cycle_B.item())

        # discriminatorの学習
        cf.set_requires_grad([D_A,D_B],True)
        optimizer_D_A.zero_grad()
        pred_real_A = D_A(real_A)
        fake_A_ = fake_A_buffer.get_images(fake_A)
        pred_fake_A = D_A(fake_A_.detach())
        loss_D_A_real = adv_loss(pred_real_A, torch.tensor(1.0).expand_as(pred_real_A).to(DEVICE))
        loss_D_A_fake = adv_loss(pred_fake_A, torch.tensor(0.0).expand_as(pred_fake_A).to(DEVICE))
        loss_D_A = (loss_D_A_fake + loss_D_A_real) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        pred_real_B = D_B(real_B)
        fake_B_ = fake_B_buffer.get_images(fake_B)
        pred_fake_B = D_B(fake_B_.detach())
        loss_D_B_real = adv_loss(pred_real_B, torch.tensor(1.0).expand_as(pred_real_B).to(DEVICE))
        loss_D_B_fake = adv_loss(pred_fake_B, torch.tensor(0.0).expand_as(pred_fake_B).to(DEVICE))
        loss_D_B = (loss_D_B_fake + loss_D_B_real) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        ll_D_A.append(loss_D_A.item())
        ll_D_B.append(loss_D_B.item())

        print(f"\r {i + 1:03} / {cf.epochSize:03} [ {n + 1:04} / {itr_size:04} ] GL A2B: {loss_G_A2B.item():.04f} B2A: {loss_G_B2A.item():.04f} DL A: {loss_D_A.item():.04f} B: {loss_D_B.item():.04f}", end = "")
        # if n == 3: break

    gab_mse = statistics.mean(ll_G_A2B)
    gba_mse = statistics.mean(ll_G_B2A)
    gca_l1 = statistics.mean(ll_G_CA)
    gcb_l1 = statistics.mean(ll_G_CB)
    da_mse = statistics.mean(ll_D_A)
    db_mse = statistics.mean(ll_D_B)
    print(f"\r {i + 1:03} / {cf.epochSize:03} [ {n + 1:04} / {itr_size:04} ] GL A2B: {gab_mse:.04f} B2A: {gba_mse:.04f} DL A: {da_mse:.04f} B: {db_mse:.04f} {time.time() - n_tm:.01f}s")
    
    # 学習の状況をCSVに保存
    with open(path_log, mode = "a") as f: print(f"{gab_mse},{gba_mse},{gca_l1},{gcb_l1},{da_mse},{db_mse}", file = f)

    # Gでの生成画像例とソース画像を連結してから保存
    buf_save_imgs = torch.cat([real_A[:min(batch_len, 32)], fake_B[:min(batch_len, 32)]], dim = 0)
    torchvision.utils.save_image(buf_save_imgs, f"{log_dir}/_e_{i + 1:03}.png", value_range=(-1.0, 1.0), normalize = True)

    # モデルの保存
    # if 0 < i and i % 10 == 0:
    #     torch.save(model_G.state_dict(), f"{log_dir}/_gen_{i:03}.pth")
    #     torch.save(model_D.state_dict(), f"{log_dir}/_dis_{i:03}.pth") # こちらは基本的には使わない

torch.save(model_G.state_dict(), f"{log_dir}/_gen_{cf.epochSize:03}.pth")
print(f"done {time.time() - s_tm:.01f}s")
