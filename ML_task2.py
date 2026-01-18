"""
植物图像分类器 - sklearn版本
使用sklearn的SVM、RandomForest、XGBoost实现更高效的分类

依赖安装:
pip install scikit-learn xgboost --break-system-packages
"""

import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入sklearn
try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    print("sklearn 已加载")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[警告] sklearn未安装，请运行: pip install scikit-learn")

# 尝试导入xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("xgboost 已加载")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] xgboost未安装，请运行: pip install xgboost")

# ==================== 配置区域 ====================
CONFIG = {
    "TRAIN_DIR": r"C:\Users\wangq\Downloads\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\train",
    "TEST_IMG_DIR": r"C:\Users\wangq\Downloads\neu-plant-seedling-classification-2025\dataset-for-task1\dataset-for-task1\test",
    "OUTPUT_CSV": r"D:\Codepython\机器学习课设\output.csv",
    
    "IMG_SIZE": (128, 128),
    "USE_AUGMENTATION": True,
    "USE_PCA": True,
    "PCA_COMPONENTS": 200,  # PCA降维后的维度
    "N_FOLDS": 5,
}
# ==================================================

PLANT_CLASSES = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']


def safe_normalize(arr):
    """安全归一化"""
    norm = np.linalg.norm(arr)
    if norm > 1e-7:
        return arr / norm
    return arr


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bovw_kmeans = None
        self.n_clusters = 100
    
    def preprocess_image(self, img):
        if img is None:
            return None
        img = cv2.resize(img, self.img_size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def segment_plant(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 15, 15])
        upper_green = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    # ========== 颜色特征 ==========
    def extract_color_histogram(self, img, bins=32):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        return np.concatenate([
            safe_normalize(hist_h.flatten()),
            safe_normalize(hist_s.flatten()),
            safe_normalize(hist_v.flatten())
        ])
    
    def extract_color_histogram_lab(self, img, bins=16):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [bins], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [bins], [0, 256])
        return np.concatenate([
            safe_normalize(hist_l.flatten()),
            safe_normalize(hist_a.flatten()),
            safe_normalize(hist_b.flatten())
        ])
    
    def extract_color_moments(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        moments = []
        for color_img in [hsv, lab]:
            for i in range(3):
                channel = color_img[:, :, i].flatten().astype(np.float64)
                mean = np.mean(channel)
                std = np.std(channel)
                skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3) if std > 0 else 0
                kurtosis = np.mean(((channel - mean) / (std + 1e-7)) ** 4) - 3 if std > 0 else 0
                moments.extend([mean / 255.0, std / 255.0, skewness / 10.0, kurtosis / 100.0])
        return np.array(moments)
    
    def extract_green_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_ranges = [
            (np.array([35, 40, 40]), np.array([85, 255, 255])),
            (np.array([25, 20, 20]), np.array([45, 255, 255])),
            (np.array([75, 20, 20]), np.array([95, 255, 255])),
        ]
        features = []
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            ratio = np.sum(mask > 0) / mask.size
            if np.sum(mask > 0) > 100:
                features.extend([
                    ratio,
                    np.mean(hsv[:, :, 0][mask > 0]) / 180.0,
                    np.mean(hsv[:, :, 1][mask > 0]) / 255.0,
                    np.mean(hsv[:, :, 2][mask > 0]) / 255.0,
                    np.std(hsv[:, :, 0][mask > 0]) / 180.0,
                    np.std(hsv[:, :, 1][mask > 0]) / 255.0
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0])
        return np.array(features)
    
    # ========== 纹理特征 ==========
    def compute_lbp(self, gray, radius=1, n_points=8):
        rows, cols = gray.shape
        lbp_img = np.zeros_like(gray)
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = gray[i, j]
                binary_code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j - radius * np.sin(angle)))
                    x = max(0, min(x, rows - 1))
                    y = max(0, min(y, cols - 1))
                    if gray[x, y] >= center:
                        binary_code |= (1 << k)
                lbp_img[i, j] = binary_code
        return lbp_img
    
    def extract_multi_scale_lbp(self, gray):
        all_hist = []
        for radius in [1, 2, 3]:
            lbp_img = self.compute_lbp(gray, radius, 8)
            hist, _ = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256))
            compressed = np.array([hist[i*8:(i+1)*8].sum() for i in range(32)])
            all_hist.append(safe_normalize(compressed.astype(np.float64)))
        return np.concatenate(all_hist)
    
    def extract_glcm_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = (gray / 16).astype(np.uint8)
        features = []
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for d in distances:
            for angle in angles:
                glcm = self._compute_glcm(gray, d, angle)
                features.extend([
                    self._glcm_contrast(glcm),
                    self._glcm_homogeneity(glcm),
                    self._glcm_energy(glcm),
                    self._glcm_correlation(glcm),
                    self._glcm_entropy(glcm)
                ])
        return np.array(features)
    
    def _compute_glcm(self, gray, distance, angle):
        levels = 16
        glcm = np.zeros((levels, levels), dtype=np.float64)
        dx = int(round(distance * np.cos(angle)))
        dy = int(round(distance * np.sin(angle)))
        rows, cols = gray.shape
        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                i2, j2 = i + dy, j + dx
                if 0 <= i2 < rows and 0 <= j2 < cols:
                    glcm[gray[i, j], gray[i2, j2]] += 1
        total = glcm.sum()
        if total > 0:
            glcm /= total
        return glcm
    
    def _glcm_contrast(self, glcm):
        levels = glcm.shape[0]
        i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing='ij')
        return np.sum((i_idx - j_idx) ** 2 * glcm) / (levels ** 2)
    
    def _glcm_homogeneity(self, glcm):
        levels = glcm.shape[0]
        i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing='ij')
        return np.sum(glcm / (1 + np.abs(i_idx - j_idx)))
    
    def _glcm_energy(self, glcm):
        return np.sum(glcm ** 2)
    
    def _glcm_correlation(self, glcm):
        levels = glcm.shape[0]
        i_idx, j_idx = np.meshgrid(range(levels), range(levels), indexing='ij')
        mu_i = np.sum(i_idx * glcm)
        mu_j = np.sum(j_idx * glcm)
        sigma_i = np.sqrt(np.sum(((i_idx - mu_i) ** 2) * glcm))
        sigma_j = np.sqrt(np.sum(((j_idx - mu_j) ** 2) * glcm))
        if sigma_i * sigma_j < 1e-7:
            return 0
        return np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm) / (sigma_i * sigma_j + 1e-7)
    
    def _glcm_entropy(self, glcm):
        glcm_nonzero = glcm[glcm > 0]
        return -np.sum(glcm_nonzero * np.log2(glcm_nonzero + 1e-10))
    
    # ========== 形状特征 ==========
    def extract_shape_features(self, img):
        mask = self.segment_plant(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros(12)
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area < 100:
            return np.zeros(12)
        
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-7)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / (h + 1e-7)
        extent = area / (w * h + 1e-7)
        
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-7)
        
        eccentricity = 0
        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                (cx, cy), (MA, ma), angle = ellipse
                eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) > 0 else 0
            except:
                pass
        
        n_defects = 0
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        if len(hull_indices) > 3 and len(largest_contour) > 3:
            try:
                defects = cv2.convexityDefects(largest_contour, hull_indices)
                if defects is not None:
                    n_defects = len([d for d in defects if d[0][3] > 2000])
            except:
                pass
        
        norm_area = area / (img.shape[0] * img.shape[1])
        
        return np.array([
            circularity, aspect_ratio, extent, solidity, eccentricity,
            norm_area, perimeter / 1000.0, n_defects / 10.0,
            w / img.shape[1], h / img.shape[0],
            len(contours) / 50.0,
            hull_area / (img.shape[0] * img.shape[1])
        ])
    
    def extract_hu_moments(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments
    
    # ========== HOG特征 ==========
    def extract_hog_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction[direction < 0] += 180
        
        cell_size = 8
        n_bins = 9
        n_cells = gray.shape[0] // cell_size
        
        hog_features = []
        for i in range(n_cells):
            for j in range(n_cells):
                cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell_dir = direction[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                hist, _ = np.histogram(cell_dir.ravel(), bins=n_bins, range=(0, 180), weights=cell_mag.ravel())
                hog_features.extend(safe_normalize(hist))
        
        return np.array(hog_features)
    
    # ========== ORB + BoVW ==========
    def extract_orb_descriptors(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return descriptors
    
    def build_bovw_vocabulary(self, all_descriptors):
        print("  构建视觉词袋词典...")
        all_desc = [desc for desc in all_descriptors if desc is not None and len(desc) > 0]
        if len(all_desc) == 0:
            return
        all_desc = np.vstack(all_desc).astype(np.float32)
        print(f"  共 {len(all_desc)} 个描述子")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(all_desc, self.n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self.bovw_kmeans = centers
        print(f"  词典大小: {self.n_clusters}")
    
    def extract_bovw_histogram(self, descriptors):
        if self.bovw_kmeans is None or descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)
        
        descriptors = descriptors.astype(np.float32)
        hist = np.zeros(self.n_clusters)
        for desc in descriptors:
            distances = np.linalg.norm(self.bovw_kmeans - desc, axis=1)
            nearest = np.argmin(distances)
            hist[nearest] += 1
        return safe_normalize(hist)
    
    # ========== 边缘特征 ==========
    def extract_edge_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        edge_mask = edges > 0
        if np.sum(edge_mask) > 0:
            hist, _ = np.histogram(direction[edge_mask], bins=12, range=(-np.pi, np.pi))
            hist = safe_normalize(hist.astype(np.float64))
        else:
            hist = np.zeros(12)
        
        return np.concatenate([
            [edge_density, np.mean(magnitude)/255.0, np.std(magnitude)/255.0],
            hist
        ])
    
    # ========== 综合特征 ==========
    def extract_all_features(self, img, include_bovw=True):
        img = self.preprocess_image(img)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        features.append(self.extract_color_histogram(img))         # 96
        features.append(self.extract_color_histogram_lab(img))     # 48
        features.append(self.extract_color_moments(img))           # 24
        features.append(self.extract_green_features(img))          # 18
        features.append(self.extract_multi_scale_lbp(gray))        # 96
        features.append(self.extract_glcm_features(img))           # 60
        features.append(self.extract_shape_features(img))          # 12
        features.append(self.extract_hu_moments(img))              # 7
        features.append(self.extract_hog_features(img))            # 576
        features.append(self.extract_edge_features(img))           # 15
        
        if include_bovw and self.bovw_kmeans is not None:
            desc = self.extract_orb_descriptors(img)
            features.append(self.extract_bovw_histogram(desc))     # 100
        
        return np.concatenate(features)


class DataAugmentor:
    @staticmethod
    def augment(img):
        augmented = [img]
        augmented.append(cv2.flip(img, 1))
        augmented.append(cv2.flip(img, 0))
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented.append(rotated)
        
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        augmented.extend([bright, dark])
        
        return augmented


def load_training_data(train_dir, extractor, augment=True):
    """加载训练数据"""
    augmentor = DataAugmentor()
    X, y = [], []
    all_descriptors = []
    
    print("\n" + "="*60)
    print("步骤1: 加载训练数据")
    print("="*60)
    
    # 第一遍：收集ORB描述子
    print("\n  收集ORB描述子...")
    for class_name in PLANT_CLASSES:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            img = cv2.imread(os.path.join(class_dir, img_file))
            if img is not None:
                img = extractor.preprocess_image(img)
                desc = extractor.extract_orb_descriptors(img)
                if desc is not None:
                    all_descriptors.append(desc)
    
    extractor.build_bovw_vocabulary(all_descriptors)
    
    # 第二遍：提取特征
    print("\n  提取特征...")
    for class_name in PLANT_CLASSES:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  [警告] 未找到: {class_dir}")
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"  {class_name}: {len(image_files)} 张图片")
        
        for img_file in image_files:
            img = cv2.imread(os.path.join(class_dir, img_file))
            if img is None:
                continue
            
            if augment:
                for aug_img in augmentor.augment(img):
                    features = extractor.extract_all_features(aug_img)
                    if features is not None:
                        X.append(features)
                        y.append(class_name)
            else:
                features = extractor.extract_all_features(img)
                if features is not None:
                    X.append(features)
                    y.append(class_name)
    
    print(f"\n  总样本数: {len(X)}, 特征维度: {len(X[0]) if X else 0}")
    return np.array(X), np.array(y)


def load_test_images(test_dir, extractor):
    """加载测试图片"""
    print("\n" + "="*60)
    print("步骤3: 加载测试数据")
    print("="*60)
    
    image_files = sorted([f for f in os.listdir(test_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    X, file_names = [], []
    for idx, img_file in enumerate(image_files):
        img = cv2.imread(os.path.join(test_dir, img_file))
        if img is not None:
            features = extractor.extract_all_features(img)
            if features is not None:
                X.append(features)
                file_names.append(img_file)
        
        if (idx + 1) % 50 == 0:
            print(f"    已处理 {idx+1}/{len(image_files)}")
    
    print(f"  成功加载 {len(X)} 张")
    return np.array(X), file_names


def build_ensemble_classifier():
    """构建集成分类器"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("需要安装sklearn: pip install scikit-learn")
    
    estimators = [
        ('svm', SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=3, 
                                      random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                          random_state=42)),
    ]
    
    if XGBOOST_AVAILABLE:
        estimators.append(
            ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                  random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
        )
    
    # 软投票集成
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    return ensemble


def main():
    print("\n" + "="*60)
    print("   植物图像分类器 - sklearn版本")
    print("   使用: SVM + RandomForest + GradientBoosting + XGBoost")
    print("="*60)
    
    if not SKLEARN_AVAILABLE:
        print("\n[错误] 请先安装sklearn: pip install scikit-learn")
        return
    
    if not os.path.exists(CONFIG["TRAIN_DIR"]):
        print(f"\n[错误] 训练目录不存在: {CONFIG['TRAIN_DIR']}")
        return
    
    # 初始化
    extractor = FeatureExtractor(img_size=CONFIG["IMG_SIZE"])
    
    # 加载数据
    X_train, y_train = load_training_data(
        CONFIG["TRAIN_DIR"], extractor, augment=CONFIG["USE_AUGMENTATION"]
    )
    
    if len(X_train) == 0:
        print("\n[错误] 未能加载训练数据!")
        return
    
    # 预处理
    print("\n" + "="*60)
    print("步骤2: 数据预处理与交叉验证")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # PCA降维
    if CONFIG["USE_PCA"]:
        print(f"\n  PCA降维: {X_scaled.shape[1]} -> {CONFIG['PCA_COMPONENTS']}")
        pca = PCA(n_components=CONFIG["PCA_COMPONENTS"], random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"  保留方差比例: {pca.explained_variance_ratio_.sum():.2%}")
    
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    
    # 交叉验证
    print("\n  5折交叉验证...")
    
    cv = StratifiedKFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=42)
    
    # 测试各个分类器
    classifiers = {
        'SVM': SVC(C=10, kernel='rbf', gamma='scale', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }
    
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                                random_state=42, use_label_encoder=False, 
                                                eval_metric='mlogloss')
    
    print("\n  各分类器交叉验证结果:")
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        print(f"    {name}: {scores.mean():.2%} (+/- {scores.std():.2%})")
    
    # 集成分类器
    ensemble = build_ensemble_classifier()
    scores = cross_val_score(ensemble, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    print(f"\n    集成分类器: {scores.mean():.2%} (+/- {scores.std():.2%})")
    
    # 训练最终模型
    print("\n" + "="*60)
    print("步骤3: 训练最终模型")
    print("="*60)
    
    print("\n  训练集成分类器...")
    ensemble.fit(X_scaled, y_encoded)
    print("  训练完成!")
    
    # 测试集预测
    if os.path.exists(CONFIG["TEST_IMG_DIR"]):
        X_test, file_names = load_test_images(CONFIG["TEST_IMG_DIR"], extractor)
        
        if len(X_test) > 0:
            print("\n" + "="*60)
            print("步骤4: 预测测试数据")
            print("="*60)
            
            X_test_scaled = scaler.transform(X_test)
            if CONFIG["USE_PCA"]:
                X_test_scaled = pca.transform(X_test_scaled)
            
            predictions_encoded = ensemble.predict(X_test_scaled)
            predictions = le.inverse_transform(predictions_encoded)
            
            result_df = pd.DataFrame({'ID': file_names, 'Category': predictions})
            result_df.to_csv(CONFIG["OUTPUT_CSV"], index=False)
            
            print(f"\n  结果已保存: {CONFIG['OUTPUT_CSV']}")
            print("\n  预测分布:")
            for cls in PLANT_CLASSES:
                count = sum(predictions == cls)
                print(f"    {cls}: {count}")
    
    print("\n" + "="*60)
    print(f"   完成! 集成分类器交叉验证准确率: {scores.mean():.2%}")
    print("="*60)


if __name__ == "__main__":
    main()