"""
import os
import torch
import torch.nn as nn
import visdom
import random

from mcnn_model import MCNN
from my_dataloader import CrowdDataset


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    vis=visdom.Visdom()
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6,
                                momentum=0.95)
    
    img_root='./data/ShanghaiTech/part_A/train_data/images'
    gt_dmap_root='./data/ShanghaiTech/part_A/train_data/ground-truth'
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

    test_img_root='./data/ShanghaiTech/part_A/test_data/images'
    test_gt_dmap_root='./data/ShanghaiTech/part_A/test_data/ground-truth'
    test_dataset=CrowdDataset(test_img_root,test_gt_dmap_root,4)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    #training phase
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    for epoch in range(0,100):

        mcnn.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            # calculate loss
            loss=criterion(et_dmap,gt_dmap)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        torch.save(mcnn.state_dict(),'./checkpoints/epoch_'+str(epoch)+".param")

        mcnn.eval()
        mae=0
        for i,(img,gt_dmap) in enumerate(test_dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap
        if mae/len(test_dataloader)<min_mae:
            min_mae=mae/len(test_dataloader)
            min_epoch=epoch
        test_error_list.append(mae/len(test_dataloader))
        print("epoch:"+str(epoch)+" error:"+str(mae/len(test_dataloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))
        vis.line(win=1,X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        vis.line(win=2,X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # show an image
        index=random.randint(0,len(test_dataloader)-1)
        img,gt_dmap=test_dataset[index]
        vis.image(win=3,img=img,opts=dict(title='img'))
        vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
        img=img.unsqueeze(0).to(device)
        gt_dmap=gt_dmap.unsqueeze(0)
        et_dmap=mcnn(img)
        et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
        vis.image(win=5,img=et_dmap/(et_dmap.max())*255,opts=dict(title='et_dmap('+str(et_dmap.sum())+')'))
        


    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))
"""

import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from mcnn_model import MCNN
from my_dataloader import CrowdDataset


if __name__ == "__main__":
    # 建議打開 cudnn，加速卷積（除非你確定有問題）
    torch.backends.cudnn.enabled = True

    # TensorBoard 記錄器
    writer = SummaryWriter(log_dir="runs/mcnn")

    # 裝置：有 GPU 用 GPU，沒有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 建立模型
    net = MCNN().to(device)

    # PyTorch 2.x 正確用法：reduction="sum" 等價於 size_average=False
    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.95)

    # ====== 路徑設定 ======
    # 注意：這裡用 ground_truth（底線），要跟你存 .npy 的資料夾一致
    img_root = "./data/ShanghaiTech/part_A/train_data/images"
    gt_dmap_root = "./data/ShanghaiTech/part_A/train_data/ground-truth"

    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    test_img_root = "./data/ShanghaiTech/part_A/test_data/images"
    test_gt_dmap_root = "./data/ShanghaiTech/part_A/test_data/ground-truth"

    test_dataset = CrowdDataset(test_img_root, test_gt_dmap_root, 4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ====== 訓練相關變數 ======
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")

    min_mae = 1e9
    best_epoch = -1

    num_epochs = 720

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0

        for i, (img, gt_dmap) in enumerate(dataloader):
            img = img.to(device)          # [B, 3, H, W]
            gt_dmap = gt_dmap.to(device)  # [B, 1, H/4, W/4] 之類

            # forward
            et_dmap = net(img)
            loss = criterion(et_dmap, gt_dmap)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / len(dataloader)

        # 存 checkpoint
        ckpt_path = f"./checkpoints/1/mcnn_epoch_{epoch}.pth"
        torch.save(net.state_dict(), ckpt_path)

        # ====== 驗證階段 ======
        net.eval()
        mae = 0.0

        with torch.no_grad():
            for i, (img, gt_dmap) in enumerate(test_dataloader):
                img = img.to(device)
                gt_dmap = gt_dmap.to(device)

                et_dmap = net(img)

                # 預測與 GT 的總人數差
                mae += torch.abs(et_dmap.sum() - gt_dmap.sum()).item()

        avg_mae = mae / len(test_dataloader)

        if avg_mae < min_mae:
            min_mae = avg_mae
            best_epoch = epoch

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test MAE: {avg_mae:.2f} | "
            f"Best MAE: {min_mae:.2f} (epoch {best_epoch})"
        )

        # ====== TensorBoard 紀錄 scalar ======
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("MAE/test", avg_mae, epoch)

        # ====== TensorBoard 顯示圖片 / GT / 預測 ======
        # 從 test set 隨機挑一張做 visualization
        idx = random.randint(0, len(test_dataset) - 1)
        img_vis, gt_vis = test_dataset[idx]  # img_vis: [3,H,W], gt_vis: [1,h,w]

        # 原始影像
        writer.add_image("Sample/image", img_vis, global_step=epoch)

        # GT density map -> normalize 到 [0,1]
        if gt_vis.max() > 0:
            gt_norm = gt_vis / gt_vis.max()
        else:
            gt_norm = gt_vis

        writer.add_image("Sample/gt_density", gt_norm, global_step=epoch)

        # 預測 density map
        img_input = img_vis.unsqueeze(0).to(device)      # [1,3,H,W]
        with torch.no_grad():
            et_vis = net(img_input).cpu().squeeze(0)     # [1,h,w] 或 [C,h,w]

        # 若是 [1,H,W]，TensorBoard 也可以當灰階圖顯示
        if et_vis.max() > 0:
            et_norm = et_vis / et_vis.max()
        else:
            et_norm = et_vis

        writer.add_image("Sample/pred_density", et_norm, global_step=epoch)

    writer.close()
    print("Training finished at:", time.strftime('%Y.%m.%d %H:%M:%S', time.localtime()))

