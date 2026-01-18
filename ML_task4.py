"""
中心凹检测 - 强化版V4 (修复版)
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import random
import math
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = "/root/data/detection"
IMG_SIZE = 512
BATCH_SIZE = 4
NUM_EPOCHS = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "/root/任务四"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class StrongAugment:
    def __call__(self, img, cx, cy, w, h):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            cx = w - cx
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            cy = h - cy
        if random.random() < 0.5:
            angle = random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            rad = math.radians(-angle)
            cx_new = (cx - w/2) * math.cos(rad) - (cy - h/2) * math.sin(rad) + w/2
            cy_new = (cx - w/2) * math.sin(rad) + (cy - h/2) * math.cos(rad) + h/2
            cx, cy = cx_new, cy_new
        if random.random() < 0.5:
            img = np.clip(img * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            img = np.clip(img.astype(np.float32) + random.randint(-30, 30), 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return img, float(np.clip(cx, 0, w)), float(np.clip(cy, 0, h))

class FoveaDataset(Dataset):
    def __init__(self, img_dir, csv_path=None, is_train=True, augment=False):
        self.img_dir = img_dir
        self.is_train = is_train
        self.aug = StrongAugment() if augment else None
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.annotations = {}
        if csv_path:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.annotations[int(row['data'])] = (float(row['Fovea_X']), float(row['Fovea_Y']))
    
    def __len__(self):
        return len(self.images)
    
    def make_heatmap(self, cx, cy, w, h, size=64, sigma=4):
        hm = np.zeros((size, size), dtype=np.float32)
        cx_s, cy_s = float(cx) * size / w, float(cy) * size / h
        x = np.arange(size, dtype=np.float32)
        y = np.arange(size, dtype=np.float32)[:, None]
        hm = np.exp(-((x - cx_s)**2 + (y - cy_s)**2) / (2 * sigma**2))
        return hm.astype(np.float32)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = int(img_name.split('.')[0])
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, img_name)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if self.is_train:
            cx, cy = self.annotations[img_id]
            if self.aug:
                img, cx, cy = self.aug(img, cx, cy, w, h)
            hm = self.make_heatmap(cx, cy, w, h)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)
            img = (img - mean) / std
            coords = torch.tensor([float(cx)/w, float(cy)/h], dtype=torch.float32)
            hm_tensor = torch.from_numpy(hm).unsqueeze(0)
            return img, coords, hm_tensor, torch.tensor([w, h], dtype=torch.float32)
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)
            img = (img - mean) / std
            return img, img_id, torch.tensor([w, h], dtype=torch.float32)

class DualHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.coord_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2), nn.Sigmoid()
        )
        self.hm_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        coord = self.coord_head(feat)
        hm = self.hm_head(feat)
        hm = nn.functional.interpolate(hm, size=(64, 64), mode='bilinear', align_corners=True)
        return coord, hm

def train():
    print(f"Device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    full_ds = FoveaDataset(os.path.join(DATA_ROOT, "train"), os.path.join(DATA_ROOT, "fovea_localization_train_GT.csv"), True, True)
    val_ds = FoveaDataset(os.path.join(DATA_ROOT, "train"), os.path.join(DATA_ROOT, "fovea_localization_train_GT.csv"), True, False)
    
    idx = list(range(80))
    random.shuffle(idx)
    train_idx, val_idx = idx[:72], idx[72:]
    
    train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    model = DualHeadNet().to(DEVICE)
    coord_loss_fn = nn.SmoothL1Loss()
    hm_loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    
    best_dist = float('inf')
    no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for imgs, coords, hms, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = imgs.to(DEVICE)
            coords = coords.to(DEVICE)
            hms = hms.to(DEVICE)
            
            optimizer.zero_grad()
            pred_coord, pred_hm = model(imgs)
            loss = coord_loss_fn(pred_coord, coords) + 0.5 * hm_loss_fn(pred_hm, hms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_dist = 0
        with torch.no_grad():
            for imgs, coords, _, sizes in val_loader:
                imgs, coords = imgs.to(DEVICE), coords.to(DEVICE)
                pred, _ = model(imgs)
                for i in range(imgs.size(0)):
                    px = pred[i,0].item() * sizes[i,0].item()
                    py = pred[i,1].item() * sizes[i,1].item()
                    gx = coords[i,0].item() * sizes[i,0].item()
                    gy = coords[i,1].item() * sizes[i,1].item()
                    val_dist += np.sqrt((px-gx)**2 + (py-gy)**2)
        val_dist /= len(val_idx)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.5f}, Dist={val_dist:.1f}px")
        
        if val_dist < best_dist:
            best_dist = val_dist
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_v4.pth"))
            print(f"  -> Saved! Best={best_dist:.1f}px")
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= 60:
            print("Early stop")
            break
    
    print("\nFine-tuning on all data...")
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_v4.pth")))
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    for ep in range(50):
        model.train()
        for imgs, coords, hms, _ in tqdm(full_loader, desc=f"FT {ep+1}/50"):
            imgs, coords, hms = imgs.to(DEVICE), coords.to(DEVICE), hms.to(DEVICE)
            optimizer.zero_grad()
            pred_coord, pred_hm = model(imgs)
            loss = coord_loss_fn(pred_coord, coords) + 0.5 * hm_loss_fn(pred_hm, hms)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_v4.pth"))
    print(f"Done! Best dist: {best_dist:.1f}px")

def inference():
    model = DualHeadNet().to(DEVICE)
    path = os.path.join(SAVE_DIR, "final_v4.pth")
    if not os.path.exists(path):
        path = os.path.join(SAVE_DIR, "best_v4.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    print(f"Loaded: {path}")
    
    mean = torch.tensor([0.485,0.456,0.406], dtype=torch.float32).view(1,3,1,1).to(DEVICE)
    std = torch.tensor([0.229,0.224,0.225], dtype=torch.float32).view(1,3,1,1).to(DEVICE)
    
    test_dir = os.path.join(DATA_ROOT, "test")
    preds = {}
    
    with torch.no_grad():
        for name in tqdm(sorted(os.listdir(test_dir)), desc="Predict"):
            if not name.endswith('.jpg'): continue
            img_id = int(name.split('.')[0])
            img = cv2.cvtColor(cv2.imread(os.path.join(test_dir, name)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            all_p = []
            for scale in [384, 448, 512, 576, 640]:
                for flip in [None, 1, 0, -1]:
                    im = img.copy()
                    if flip is not None:
                        im = cv2.flip(im, flip)
                    im = cv2.resize(im, (scale, scale))
                    t = torch.from_numpy(im.transpose(2,0,1).astype(np.float32)).unsqueeze(0)/255.0
                    t = nn.functional.interpolate((t.to(DEVICE)-mean)/std, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=True)
                    p, hm = model(t)
                    p = p[0].cpu().numpy()
                    
                    hm_np = hm[0,0].cpu().numpy()
                    hy, hx = np.unravel_index(np.argmax(hm_np), hm_np.shape)
                    hm_coord = np.array([hx/64.0, hy/64.0], dtype=np.float32)
                    
                    p = 0.7 * p + 0.3 * hm_coord
                    
                    if flip == 1 or flip == -1:
                        p[0] = 1 - p[0]
                    if flip == 0 or flip == -1:
                        p[1] = 1 - p[1]
                    all_p.append(p)
            
            avg = np.mean(all_p, axis=0)
            preds[img_id] = (float(avg[0]*w), float(avg[1]*h))
    
    rows = []
    for img_id in sorted(preds.keys()):
        x, y = preds[img_id]
        rows.append({'ImageID': f"{img_id}_Fovea_X", 'value': x})
        rows.append({'ImageID': f"{img_id}_Fovea_Y", 'value': y})
    
    out_path = os.path.join(SAVE_DIR, "submission_v4.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode == 'train': train()
    elif mode == 'test': inference()
    else: train(); inference()
