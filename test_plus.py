#%%
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM

from mcnn_plus import MCNNPlus
from my_dataloader import CrowdDataset

def visualize_result(img_root, gt_dmap_root, model_param_path, index):
    """
    Generate a figure with:
    1. Input Image
    2. Ground Truth Density Map
    3. Predicted Density Map (MCNN)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    mcnn = MCNNPlus().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    mcnn.eval()

    # Load dataset
    dataset = CrowdDataset(img_root, gt_dmap_root, 4)
    img, gt_dmap = dataset[index]   # img: [3,H,W], gt_dmap: [1,h,w]

    # Prepare image for model
    img_input = img.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred_dmap = mcnn(img_input).cpu().squeeze(0).squeeze(0).numpy()

    # GT to numpy
    gt_dmap_np = gt_dmap.squeeze(0).numpy()

    # --- Normalize for visualization ---
    if gt_dmap_np.max() > 0:
        gt_vis = gt_dmap_np / gt_dmap_np.max()
    else:
        gt_vis = gt_dmap_np

    if pred_dmap.max() > 0:
        pred_vis = pred_dmap / pred_dmap.max()
    else:
        pred_vis = pred_dmap

    # --- Input image from tensor to numpy ---
    img_np = img.permute(1, 2, 0).numpy()
    img_np = img_np - img_np.min()
    img_np = img_np / img_np.max()

    # --- Plot figure ---
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title(f"GT Density (sum={gt_dmap_np.sum():.1f})")
    plt.imshow(gt_vis, cmap=CM.jet)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Predicted Density (sum={pred_dmap.sum():.1f})")
    plt.imshow(pred_vis, cmap=CM.jet)
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    mcnn=MCNNPlus().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    mcnn=MCNNPlus().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='./data/ShanghaiTech/part_A/test_data/images'
    gt_dmap_root='./data/ShanghaiTech/part_A/test_data/ground-truth'
    model_param_path='./checkpoints/2/mcnn_epoch_642.pth'
    cal_mae(img_root,gt_dmap_root,model_param_path)
    estimate_density_map(img_root,gt_dmap_root,model_param_path,3) 
    visualize_result(img_root, gt_dmap_root, model_param_path, index=3)
    