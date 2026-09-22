import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from DSRNet import DSRNet


parser = argparse.ArgumentParser("DSRNet")
parser.add_argument("--hiera_path", type=str, required=True, 
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--dinov2_path", type=str, required=True, 
                    help="path to the pretrained dinov2")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20, 
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_workers", default=0, type=int,
                    help="data loading workers; 0 avoids Windows subprocess overhead")
parser.add_argument("--image_size", default=352, type=int,
                    help="square image and mask size after preprocessing")
parser.add_argument("--weight_decay", default=5e-4, type=float)


# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()


def compute_shape_field(gt_mask):
    """根据GT计算法向场 s_gt"""
    gx = F.conv2d(
        gt_mask,
        torch.tensor(
            [[[[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]]],
            device=gt_mask.device,
            dtype=gt_mask.dtype
        ),
        padding=1
    )
    gy = F.conv2d(
        gt_mask,
        torch.tensor(
            [[[[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]]],
            device=gt_mask.device,
            dtype=gt_mask.dtype
        ),
        padding=1
    )
    s = torch.cat([gx, gy], dim=1)
    s = F.normalize(s, dim=1)
    return s
class CCSShapeLoss(nn.Module):
    def __init__(self, lambda_shape=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_shape = lambda_shape

        self.sobel_x = torch.tensor(
            [[[[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]]],
            dtype=torch.float32
        )
        self.sobel_y = torch.tensor(
            [[[[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]]],
            dtype=torch.float32
        )

    def forward(self, logits, mask):
        L_bce = self.bce(logits, mask)

        prob = torch.sigmoid(logits)

        sobel_x = self.sobel_x.to(prob.device)
        sobel_y = self.sobel_y.to(prob.device)

        gx = F.conv2d(prob, sobel_x, padding=1)
        gy = F.conv2d(prob, sobel_y, padding=1)

        grad = torch.cat([gx, gy], dim=1)
        grad = F.normalize(grad, dim=1, eps=1e-6)

        s_gt = compute_shape_field(mask)
        s_gt = F.normalize(s_gt, dim=1, eps=1e-6)

        L_shape = F.relu(-(grad * s_gt).sum(1)).mean()

        return L_bce + self.lambda_shape * L_shape


def main(args):    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run in the sam environment with a CUDA-enabled PyTorch installation.")
    dataset = FullDataset(args.train_image_path, args.train_mask_path, args.image_size, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)
    device = torch.device("cuda")
    model = SAM2UNeXT(args.hiera_path, args.dinov2_path)
    model.to(device)
    print(f"Training device: {next(model.parameters()).device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Input: {args.image_size}x{args.image_size}; batch size: {args.batch_size}; workers: {args.num_workers}", flush=True)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = CCSShapeLoss(pred, target)
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print(f"epoch:{epoch + 1}-{i + 1}: loss:{loss.item():.6f}; "
                      f"prediction device:{pred.device}; "
                      f"CUDA allocated:{torch.cuda.memory_allocated() / 1024**3:.2f} GiB", flush=True)
                
        scheduler.step()
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNeXT-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNeXT-%d.pth'% (epoch + 1)))


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    args = parser.parse_args()
    main(args)
