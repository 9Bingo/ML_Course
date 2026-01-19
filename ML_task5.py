import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
class Config:
    ROOT_DIR = r'/home/wuyuzhang/litianyuan/ML5/segmentation'
    TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
    TEST_DIR = os.path.join(ROOT_DIR, 'test/image')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'image')
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    PRETRAINED_PATH = os.path.join(ROOT_DIR, 'pretrained/resnet34.pth')
       
    # 注意：因为我们下面要修改输入图片的颜色通道（只用绿色通道增强），
    # 所以这里的 Mean/Std 其实不再适用 RGB 的统计值。
    # 我们可以简单地设为 0.5/0.5，让 BatchNorm 去自适应，或者重新计算。
    # 这里为了简便，使用 0.5
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    IMAGE_SIZE = 512
    BATCH_SIZE = 4 
    NUM_EPOCHS = 60 # 增加轮数，因为 strong augmentation 需要更久收敛
    LEARNING_RATE = 2e-4
    NUM_FOLDS = 5
    
    ENCODER = 'resnet34'
    MODEL_ARCH = 'UnetPlusPlus'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 2024

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)

# ==================== 进阶数据处理 (关键优化) ====================
def preprocess_fundus(image):
    """
    专门针对眼底图像的预处理：
    1. 提取绿色通道 (G channel) - 血管对比度最高
    2. 应用 CLAHE 增强对比度
    3. 将单通道 G 复制 3 次变成 3 通道 (为了喂给 ResNet)
    """
    # OpenCV 读入是 BGR，split 后顺序是 b, g, r
    b, g, r = cv2.split(image)
    
    # 创建 CLAHE 对象 (Clip Limit 设为 2.0 到 4.0 之间)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 只增强绿色通道
    enhanced_g = clahe.apply(g)
    
    # 方案 A: 只用增强后的绿色通道，堆叠3次 (效果通常最好)
    merged = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    
    # 方案 B (备选): 保留色彩信息，但增强 G (如果方案A效果不好可以试这个)
    # enhanced_l = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:,:,0])
    # ...
    
    return merged

class VesselDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_test = (mask_paths is None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图片
        image = cv2.imread(str(self.image_paths[idx]))
        
        # [优化点 1] 应用眼底图像特有的预处理
        image = preprocess_fundus(image)
        # 注意：preprocess_fundus 返回的是 BGR/Grayscale 格式，不需要再转 RGB，
        # 因为我们是把灰度图堆叠了3次，RGB/BGR 是一样的。

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0)
        return image, mask

# ==================== 增强 (加入弹性形变) ====================
def get_train_transforms():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # [优化点 2] 几何增强：模拟眼球的曲率和血管扭曲
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.3), # 30% 的概率应用
        
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        
        # 亮度增强不需要太强了，因为我们已经做了 CLAHE
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MotionBlur(p=0.2),
        ], p=0.3),
        
        A.Normalize(mean=Config.MEAN, std=Config.STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=Config.MEAN, std=Config.STD),
        ToTensorV2(),
    ])

# ==================== 损失函数 (引入 Tversky) ====================
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1):
        super().__init__()
        self.alpha = alpha # False Positive 权重
        self.beta = beta   # False Negative 权重 (设高一点以减少漏检)
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        return 1 - Tversky

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # [优化点 3] 使用 Tversky Loss 替代纯 Dice
        # beta=0.7 意味着我们更痛恨“漏检” (Recall更重要)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        return self.tversky(pred, target) * 0.7 + self.bce(pred, target) * 0.3

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

# ==================== 训练流程 ====================
def train_kfold():
    print("Step 1: Training (Optimized for Vessels)...")
    
    image_dir = Path(Config.TRAIN_DIR) / 'image'
    label_dir = Path(Config.TRAIN_DIR) / 'label'
    image_paths = sorted(list(image_dir.glob('*.*')))
    mask_paths = [label_dir / img.name for img in image_paths]
    
    valid = [(i, m) for i, m in zip(image_paths, mask_paths) if m.exists()]
    if not valid:
        print("Error: No paired image/mask found. Check paths.")
        return
    image_paths, mask_paths = zip(*valid)
    image_paths, mask_paths = list(image_paths), list(mask_paths)

    kfold = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
        print(f"\nFold {fold+1}/{Config.NUM_FOLDS}")
        
        train_ds = VesselDataset([image_paths[i] for i in train_idx], [mask_paths[i] for i in train_idx], get_train_transforms())
        val_ds = VesselDataset([image_paths[i] for i in val_idx], [mask_paths[i] for i in val_idx], get_val_transforms())
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = smp.UnetPlusPlus(encoder_name=Config.ENCODER, encoder_weights=None, in_channels=3, classes=1)
        
        if os.path.exists(Config.PRETRAINED_PATH):
            state_dict = torch.load(Config.PRETRAINED_PATH)
            try:
                model.encoder.load_state_dict(state_dict)
            except:
                model.encoder.load_state_dict(state_dict, strict=False)
        else:
             print(f"WARNING: No pretrained weights at {Config.PRETRAINED_PATH}")

        model = model.to(Config.DEVICE)
        
        # 学习率稍微调低一点，因为用了强增强
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE * 0.8, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
        criterion = CombinedLoss()
        
        best_loss = float('inf')
        
        for epoch in range(Config.NUM_EPOCHS):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
            scheduler.step()
            
            if loss < best_loss:
                best_loss = loss
                torch.save({'model': model.state_dict()}, os.path.join(Config.MODEL_DIR, f'fold{fold}_best.pth'))
        
        print(f"Fold {fold+1} Finished. Best Loss: {best_loss:.4f}")

# ==================== 预测与后处理 ====================
def remove_small_objects(mask, min_size=30):
    binary = (mask == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    new_mask = np.ones_like(mask) * 255
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 0
    return new_mask.astype(np.uint8)

def predict_ensemble():
    print("\nStep 2: Predicting...")
    test_dir = Path(Config.TEST_DIR)
    test_images = sorted(list(test_dir.glob('*.*')))
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    models = []
    for fold in range(Config.NUM_FOLDS):
        path = os.path.join(Config.MODEL_DIR, f'fold{fold}_best.pth')
        if os.path.exists(path):
            model = smp.UnetPlusPlus(encoder_name=Config.ENCODER, encoder_weights=None, in_channels=3, classes=1)
            model.load_state_dict(torch.load(path)['model'])
            model.to(Config.DEVICE).eval()
            models.append(model)
    
    if not models:
        print("No models trained!")
        return

    print(f"Loaded {len(models)} models for ensemble.")
    val_transform = get_val_transforms()

    for img_path in tqdm(test_images):
        original_image = cv2.imread(str(img_path))
        if original_image is None: continue
        
        # [关键] 预测时也要用同样的预处理 (Green channel + CLAHE)
        image = preprocess_fundus(original_image)
        
        aug = val_transform(image=image)
        img_tensor = aug['image'].unsqueeze(0).to(Config.DEVICE)
        
        preds = []
        with torch.no_grad():
            for model in models:
                # TTA 
                p1 = torch.sigmoid(model(img_tensor))
                p2 = torch.sigmoid(model(torch.flip(img_tensor, [3]))) # H Flip
                p3 = torch.sigmoid(model(torch.flip(img_tensor, [2]))) # V Flip
                # 还可以加一个 Transpose
                p4 = torch.sigmoid(model(torch.transpose(img_tensor, 2, 3)))
                
                # 恢复 p4 需要转置回来
                p4_back = torch.transpose(p4, 2, 3)
                
                pred_tta = (p1 + torch.flip(p2, [3]) + torch.flip(p3, [2]) + p4_back) / 4.0
                preds.append(pred_tta)
        
        pred = torch.mean(torch.stack(preds), dim=0)[0, 0].cpu().numpy()
        
        # [优化点 4] 阈值微调
        # 默认 0.5。为了捕获更多细小血管，可以稍微降低阈值，例如 0.45 或 0.4
        # 但是降低阈值会增加噪点，所以要配合 remove_small_objects
        threshold = 0.42 
        
        final_output = np.where(pred > threshold, 0, 255).astype(np.uint8)
        final_output = cv2.resize(final_output, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 适当增大去噪力度，因为降低了阈值
        final_output = remove_small_objects(final_output, min_size=30)
        
        save_name = img_path.stem + ".png"
        save_path = str(Path(Config.OUTPUT_DIR) / save_name)
        cv2.imwrite(save_path, final_output)

    print(f"Done. Saved to {Config.OUTPUT_DIR}")

if __name__ == '__main__':
    train_kfold()
    predict_ensemble()
