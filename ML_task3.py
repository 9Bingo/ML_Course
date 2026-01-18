"""
面部表情分类 v9 - 3模型集成 (目标70%+)
训练3个不同架构的模型，集成预测
"""

import os, random, warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ========================== 配置 ==========================
class Config:
    IMG_SIZE = 72  # 稍大一点
    BATCH_SIZE = 64
    NUM_EPOCHS = 60
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========================== 模块 ==========================
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch//r), nn.ReLU(), nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x):
        b,c,_,_ = x.size()
        y = x.view(b,c,-1).mean(2)
        return x * self.fc(y).view(b,c,1,1)

class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch,ch//r), nn.ReLU(), nn.Linear(ch//r,ch), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2,1,7,padding=3), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x).unsqueeze(-1).unsqueeze(-1)
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        return x * self.sa(torch.cat([avg_out, max_out], dim=1))

# ========================== 模型1: SE-ResNet ==========================
class ResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)) if stride!=1 or in_ch!=out_ch else nn.Identity()
    def forward(self, x):
        return F.relu(self.se(self.conv(x)) + self.shortcut(x))

class Model1_SEResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1,64,3,1,1,bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = nn.Sequential(ResBlockSE(64,64), ResBlockSE(64,64))
        self.layer2 = nn.Sequential(ResBlockSE(64,128,2), ResBlockSE(128,128))
        self.layer3 = nn.Sequential(ResBlockSE(128,256,2), ResBlockSE(256,256))
        self.layer4 = nn.Sequential(ResBlockSE(256,512,2), ResBlockSE(512,512))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)

# ========================== 模型2: VGG-style + CBAM ==========================
class Model2_VGG(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.cbam = CBAM(512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        return self.head(x)

# ========================== 模型3: Inception-style ==========================
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, mid, 1), nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, mid, 1), nn.Conv2d(mid, mid, 5, padding=2), nn.BatchNorm2d(mid), nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU())
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class Model3_Inception(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.inc1 = InceptionBlock(64, 128)
        self.inc2 = InceptionBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.inc3 = InceptionBlock(256, 512)
        self.inc4 = InceptionBlock(512, 512)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.inc1(x))
        x = self.pool(self.inc2(x))
        x = self.inc3(x)
        x = self.inc4(x)
        return self.head(x)

# ========================== 数据 ==========================
class FERDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        if self.transform: img = self.transform(img)
        return (img, self.labels[idx]) if self.labels is not None else img

def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1), shear=5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

# ========================== 训练单个模型 ==========================
def train_model(model, tr_loader, val_loader, model_name, epochs=60):
    print(f"\n{'='*50}")
    print(f"训练 {model_name}")
    print(f"{'='*50}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_acc = 0
    save_path = f'/root/model_{model_name}.pth'
    
    for ep in range(epochs):
        # Train
        model.train()
        correct, total = 0, 0
        pbar = tqdm(tr_loader, desc=f'Ep{ep+1}', leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        tr_acc = 100.*correct/total
        
        # Val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
                pred = model(imgs).argmax(1)
                total += labels.size(0)
                correct += (pred==labels).sum().item()
        val_acc = 100.*correct/total
        scheduler.step()
        
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            mark = " ✅"
        
        if (ep+1) % 5 == 0 or mark:
            print(f"Epoch {ep+1}: Train {tr_acc:.1f}%, Val {val_acc:.1f}%{mark}")
    
    print(f"🏆 {model_name} 最佳: {best_acc:.2f}%")
    model.load_state_dict(torch.load(save_path))
    return model, best_acc

# ========================== TTA ==========================
def get_tta_transforms(img_size):
    return [
        transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]),
        transforms.Compose([transforms.Resize((img_size,img_size)), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]),
        transforms.Compose([transforms.Resize((img_size+8,img_size+8)), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]),
        transforms.Compose([transforms.Resize((img_size+8,img_size+8)), transforms.CenterCrop(img_size), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]),
    ]

def predict_ensemble_tta(models, test_paths, device):
    """集成多个模型 + TTA"""
    for m in models: m.eval()
    preds = []
    tta_transforms = get_tta_transforms(Config.IMG_SIZE)
    
    with torch.no_grad():
        for path in tqdm(test_paths, desc='集成TTA预测'):
            img = Image.open(path).convert('L')
            
            # 所有模型所有TTA的概率累加
            total_prob = None
            for model in models:
                for t in tta_transforms:
                    x = t(img).unsqueeze(0).to(device)
                    p = F.softmax(model(x), dim=1)
                    total_prob = p if total_prob is None else total_prob + p
            
            total_prob /= (len(models) * len(tta_transforms))
            preds.append(total_prob.argmax(1).item())
    
    return preds

# ========================== 主程序 ==========================
def main():
    print("=" * 60)
    print("⚡ 面部表情分类 v9 - 3模型集成 (目标70%+)")
    print("=" * 60)
    
    set_seed(Config.SEED)
    print(f"设备: {Config.DEVICE}")
    
    # 加载数据
    train_dir, test_dir = "/root/fer_data/train", "/root/fer_data/test"
    label_map = {f: i for i, f in enumerate(sorted(os.listdir(train_dir)))}
    print(f"标签映射: {label_map}")
    
    paths, labels = [], []
    for emo, lbl in label_map.items():
        folder = os.path.join(train_dir, emo)
        for img in os.listdir(folder):
            if img.lower().endswith(('.png','.jpg','.jpeg')):
                paths.append(os.path.join(folder, img))
                labels.append(lbl)
    
    test_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
    
    print(f"训练: {len(paths)}, 测试: {len(test_paths)}")
    
    # 划分
    tr_idx, val_idx = train_test_split(range(len(paths)), test_size=0.1, stratify=labels, random_state=Config.SEED)
    tr_paths, tr_labels = [paths[i] for i in tr_idx], [labels[i] for i in tr_idx]
    val_paths, val_labels = [paths[i] for i in val_idx], [labels[i] for i in val_idx]
    
    # 平衡采样
    class_counts = Counter(tr_labels)
    sampler = WeightedRandomSampler([1.0/class_counts[l] for l in tr_labels], len(tr_labels))
    
    # DataLoader
    train_tf = get_transforms(Config.IMG_SIZE, is_train=True)
    val_tf = get_transforms(Config.IMG_SIZE, is_train=False)
    tr_loader = DataLoader(FERDataset(tr_paths, tr_labels, train_tf), batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(FERDataset(val_paths, val_labels, val_tf), batch_size=Config.BATCH_SIZE, num_workers=4, pin_memory=True)
    
    # 训练3个模型
    models = []
    accs = []
    
    # 模型1: SE-ResNet
    m1 = Model1_SEResNet(len(label_map)).to(Config.DEVICE)
    m1, acc1 = train_model(m1, tr_loader, val_loader, "SE-ResNet", epochs=Config.NUM_EPOCHS)
    models.append(m1); accs.append(acc1)
    
    # 模型2: VGG-CBAM
    set_seed(Config.SEED + 1)  # 不同随机种子
    m2 = Model2_VGG(len(label_map)).to(Config.DEVICE)
    m2, acc2 = train_model(m2, tr_loader, val_loader, "VGG-CBAM", epochs=Config.NUM_EPOCHS)
    models.append(m2); accs.append(acc2)
    
    # 模型3: Inception
    set_seed(Config.SEED + 2)
    m3 = Model3_Inception(len(label_map)).to(Config.DEVICE)
    m3, acc3 = train_model(m3, tr_loader, val_loader, "Inception", epochs=Config.NUM_EPOCHS)
    models.append(m3); accs.append(acc3)
    
    print(f"\n{'='*50}")
    print(f"单模型准确率: {[f'{a:.1f}%' for a in accs]}")
    print(f"平均: {np.mean(accs):.2f}%")
    print(f"{'='*50}")
    
    # 集成预测
    print("\n集成TTA预测...")
    preds = predict_ensemble_tta(models, test_paths, Config.DEVICE)
    
    # 修正标签映射
    label_fix = {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4}
    preds_fixed = [label_fix[p] for p in preds]
    
    # 保存
    sub = pd.DataFrame({
        'ID': [f"{id}.jpg" for id in test_ids],
        'Emotion': preds_fixed
    })
    sub.to_csv('/root/submission.csv', index=False)
    print(f"\n✅ 保存至: /root/submission.csv")
    
    print("\n预测分布:")
    for l, c in sorted(Counter(preds_fixed).items()):
        print(f"  {l}: {c}")

if __name__ == '__main__':
    main()
