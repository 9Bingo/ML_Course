# # import os
# # import glob
# # import cv2
# # import numpy as np
# # import pandas as pd
# # import mahotas  # 用于提取纹理特征
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.svm import SVC
# # from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# # from sklearn.metrics import f1_score, classification_report
# # from xgboost import XGBClassifier


# # # =================配置路径=================
# # # 请根据你的实际文件夹路径修改这里   
# # TRAIN_PATH = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\train"  # 训练集文件夹路径，里面应该是分文件夹存放或者直接是图片
# # TEST_PATH = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\test"    # 测试集文件夹路径  
# # # 假设训练集结构是 train/类别名/图片.png，如果不一致请在读取部分调整

# # # 定义固定的特征提取参数
# # # FIXED_SIZE = (500, 500) # 统一图片大小，方便处理
# # BINS = 8 # 直方图的柱数

# # # =================第一步：特征提取函数工具箱=================

# # def rgb_bgr_to_hsv_mask(image):
# #     """
# #     图像分割：利用HSV颜色空间提取绿色植物，去除背景
# #     """
# #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
# #     # 定义绿色的范围 (根据实际数据集可能需要微调)
# #     # 这是一个比较通用的植物绿色范围
# #     lower_green = np.array([25, 52, 72])
# #     upper_green = np.array([102, 255, 255])
    
# #     mask = cv2.inRange(hsv, lower_green, upper_green)

# #     # 使用位运算保留原图中的绿色部分，背景变黑
# #     result = cv2.bitwise_and(image, image, mask=mask)
# #     return result, mask

# # def fd_hu_moments(image):
# #     """
# #     提取形状特征：Hu矩 (Hu Moments)
# #     关注植物叶片的轮廓形状
# #     """
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     feature = cv2.HuMoments(cv2.moments(image)).flatten()
# #     return feature

# # def fd_haralick(image):
# #     """
# #     提取纹理特征：Haralick Texture
# #     关注叶片的纹理细节
# #     """
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     # 计算 Haralick 纹理特征，取平均值作为最终向量
# #     haralick = mahotas.features.haralick(gray).mean(axis=0)
# #     return haralick

# # def fd_histogram(image, mask=None):
# #     """
# #     提取颜色特征：颜色直方图
# #     """
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #     # 计算直方图，只计算Mask区域（即植物区域）
# #     hist  = cv2.calcHist([image], [0, 1, 2], mask, [BINS, BINS, BINS], [0, 256, 0, 256, 0, 256])
# #     cv2.normalize(hist, hist)
# #     return hist.flatten()

# # # =================第二步：数据加载与特征构建=================

# # # def extract_global_features(image_path):
# # #     """
# # #     主特征提取逻辑：读取图片 -> 预处理 -> 提取三种特征 -> 拼接
# # #     """
# # #     image = cv2.imread(image_path)
# # #     image = cv2.resize(image, FIXED_SIZE)
    
# # #     # 1. 分割（去背景）
# # #     masked_image, mask = rgb_bgr_to_hsv_mask(image)
    
# # #     # 2. 提取特征
# # #     fv_hu_moments = fd_hu_moments(masked_image)
# # #     fv_haralick   = fd_haralick(masked_image)
# # #     fv_histogram  = fd_histogram(masked_image, mask)
    
# # #     # 3. 拼接成一个长向量
# # #     global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    
# # #     return global_feature

# # # 新增一个提取颜色统计信息的函数
# # def fd_color_stats(image, mask=None):
# #     """
# #     提取颜色通道的统计特征：均值(Mean)和标准差(Std)
# #     这比直方图更直接地反应了叶子是"深绿"还是"浅绿"
# #     """
# #     # 将图片转换到 LAB 空间 (更符合人类视觉感知)
# #     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# #     l, a, b = cv2.split(lab)
    
# #     # 只计算 mask 区域（即叶子区域）的统计值
# #     masked_l = l[mask > 0]
# #     masked_a = a[mask > 0]
# #     masked_b = b[mask > 0]

# #     if len(masked_l) == 0:
# #         return np.zeros(6)

# #     # 计算均值和标准差
# #     features = [
# #         np.mean(masked_l), np.std(masked_l),
# #         np.mean(masked_a), np.std(masked_a),
# #         np.mean(masked_b), np.std(masked_b)
# #     ]
# #     return np.array(features)


# # def resize_image(image, max_size=(800, 800)):
# #     """
# #     按照最大尺寸调整图像大小，保持原始比例
# #     """
# #     h, w = image.shape[:2]
# #     ratio = min(max_size[0] / h, max_size[1] / w)  # 计算保持比例的缩放因子
# #     new_dims = (int(w * ratio), int(h * ratio))  # 计算新尺寸
# #     resized_image = cv2.resize(image, new_dims)  # 执行调整尺寸
# #     return resized_image



# # # 修改主提取函数
# # def extract_global_features(image_path):
# #     image = cv2.imread(image_path)
# #     # image = cv2.resize(image, FIXED_SIZE)
# #     image = resize_image(image, max_size=(800, 800))  # 使用较大的自适应大小

# #     # 1. 分割
# #     masked_image, mask = rgb_bgr_to_hsv_mask(image)
    
# #     # 2. 提取特征
# #     fv_hu_moments = fd_hu_moments(masked_image)
# #     fv_haralick   = fd_haralick(masked_image)
# #     fv_histogram  = fd_histogram(masked_image, mask)
    
# #     # === 新增：颜色统计特征 ===
# #     fv_color_stats = fd_color_stats(image, mask)
    
# #     # 3. 拼接
# #     global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_color_stats])
    
# #     return global_feature


# # print("开始处理训练集数据...")

# # # 准备存储列表
# # global_features = []
# # labels = []

# # # 假设训练集文件夹结构如下：
# # # train_dir/
# # #   ├── Black-grass/
# # #   ├── Common wheat/
# # #   ...
# # # 如果你的训练集是所有图片混在一起，文件名里没有标签，需要根据提供的 train.csv 或文件夹名来做。
# # # 这里假设是按文件夹分类的标准 ImageNet 格式。
# # # **注意：如果你的数据是平铺的（所有图片在一个文件夹），请读取 train.csv 获取 label**

# # # 这里根据你的描述 "该数据集包含 5种植物，每种植物数量为100张图片"，
# # # 假设它们在不同的文件夹里，或者你需要先读取一个 train.csv。
# # # 下面代码假设是分文件夹的，如果是csv读取，请替换这部分循环。

# # train_categories = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
# # # 假设 train_path 下面有这5个文件夹
# # TRAIN_DIR = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\train"

# # # 模拟数据读取过程 (请根据实际文件结构修改!)
# # # 示例：如果是文件夹结构
# # for category in train_categories:
# #     path = os.path.join(TRAIN_DIR, category)
# #     # 获取该类别下所有图片
# #     # 根据实际图片后缀修改 *.png 或 *.jpg
# #     images = glob.glob(os.path.join(path, "*.png")) 
# #     print(f"正在处理类别: {category}, 数量: {len(images)}")
    
# #     for image_file in images:
# #         # 提取特征
# #         feature = extract_global_features(image_file)
# #         global_features.append(feature)
# #         labels.append(category)

# # print(f"特征提取完成。特征向量维度: {np.array(global_features).shape}")

# # # =================第三步：模型训练与调优=================

# # # 编码标签 (String -> Int)
# # target_names = np.unique(labels)
# # le = LabelEncoder()
# # target = le.fit_transform(labels)

# # # 数据标准化 (非常重要！)
# # scaler = StandardScaler()
# # global_features_scaled = scaler.fit_transform(np.array(global_features))

# # print("开始训练模型...")

# # # # 方案A：使用随机森林 (通常鲁棒性好，不容易过拟合)
# # # # 方案B：使用 SVM (在小样本、高维特征上表现极佳)
# # # # 为了得高分，我们使用 SVM 并配合 GridSearchCV 寻找最优参数

# # # # 定义参数网格
# # # param_grid = {
# # #     'C': [1, 10, 100, 1000],
# # #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
# # #     'kernel': ['rbf', 'linear']
# # # }

# # # model = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=2, cv=5, scoring='f1_weighted')
# # # model.fit(global_features_scaled, target)

# # # print(f"最佳参数: {model.best_params_}")
# # # print(f"最佳CV分数: {model.best_score_}")

# # print("开始训练 XGBoost 模型...")

# # # 定义 XGBoost 模型
# # # n_estimators: 树的数量
# # # learning_rate: 学习率
# # # max_depth: 树的深度
# # xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)

# # # 定义搜索参数
# # param_grid = {
# #     'n_estimators': [100, 200, 500],
# #     'learning_rate': [0.01, 0.05, 0.1],
# #     'max_depth': [3, 5, 7],
# #     'subsample': [0.8, 1.0]
# # }

# # # 使用 GridSearch 搜索最佳参数
# # model = GridSearchCV(xgb, param_grid, refit=True, verbose=1, cv=5, scoring='f1_weighted')
# # model.fit(global_features_scaled, target)

# # print(f"最佳参数: {model.best_params_}")
# # print(f"最佳CV分数: {model.best_score_}") 

# # # =================第四步：预测测试集=================

# # print("开始处理测试集并生成提交文件...")

# # # 读取 sample_submission 或者是 test.csv 来获取需要预测的文件名列表
# # # 根据你提供的附件，文件名在 ID 列
# # sample_submission = pd.read_csv(r"C:\Users\12132\Desktop\ML1\submission-for-task1.csv") # 读取你上传的那个文件作为模板
# # test_ids = sample_submission['ID'].values

# # test_features = []
# # valid_ids = []

# # for filename in test_ids:
# #     # 拼接测试集完整路径
# #     img_path = os.path.join(TEST_PATH, filename)
    
# #     if os.path.exists(img_path):
# #         feature = extract_global_features(img_path)
# #         test_features.append(feature)
# #         valid_ids.append(filename)
# #     else:
# #         print(f"警告: 找不到文件 {filename}")
# #         # 如果找不到文件，可能需要填充零向量或跳过，视情况而定
# #         # 这里假设都能找到

# # # 标准化测试集特征 (使用训练集的scaler)
# # test_features_scaled = scaler.transform(np.array(test_features))

# # # 预测
# # predictions = model.predict(test_features_scaled)
# # predicted_labels = le.inverse_transform(predictions)

# # # =================第五步：保存结果=================

# # submission_df = pd.DataFrame({
# #     'ID': valid_ids,
# #     'Category': predicted_labels
# # })

# # # 保存为 csv
# # submission_df.to_csv("submission_result.csv", index=False)
# # print("结果已保存至 submission_result.csv")


# import os
# import glob
# import cv2
# import numpy as np
# import pandas as pd
# import mahotas
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, StratifiedKFold

# # =================配置路径=================
# TRAIN_PATH = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\train"
# TEST_PATH = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\test"    
# SUBMISSION_TEMPLATE = r"C:\Users\12132\Desktop\ML1\submission-for-task1.csv"

# BINS = 8 

# # =================第一步：特征提取工具箱=================

# def resize_image(image, max_size=(800, 800)):
#     """ 保留你的比例缩放函数 """
#     h, w = image.shape[:2]
#     ratio = min(max_size[0] / h, max_size[1] / w)
#     new_dims = (int(w * ratio), int(h * ratio))
#     return cv2.resize(image, new_dims)

# def rgb_bgr_to_hsv_mask(image):
#     """
#     修正：减小形态学操作的力度，防止把细长的草叶抹掉
#     """
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # 稍微放宽一点绿色的范围
#     lower_green = np.array([25, 40, 40]) 
#     upper_green = np.array([100, 255, 255])
    
#     mask = cv2.inRange(hsv, lower_green, upper_green)
    
#     # === 修正：使用很小的核，只去像素级噪点，保留细草 ===
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # 从11改回3
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
#     result = cv2.bitwise_and(image, image, mask=mask)
#     return result, mask

# def fd_hu_moments(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     feature = cv2.HuMoments(cv2.moments(image)).flatten()
#     return feature

# def fd_haralick(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     try:
#         haralick = mahotas.features.haralick(gray).mean(axis=0)
#     except ValueError:
#         return np.zeros(13)
#     return haralick

# def fd_histogram(image, mask=None):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist  = cv2.calcHist([image], [0, 1, 2], mask, [BINS, BINS, BINS], [0, 256, 0, 256, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# def fd_color_stats(image, mask=None):
#     # 颜色统计特征非常有用，保留
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     masked_l = l[mask > 0]
#     masked_a = a[mask > 0]
#     masked_b = b[mask > 0]

#     if len(masked_l) == 0:
#         return np.zeros(6)

#     features = [
#         np.mean(masked_l), np.std(masked_l),
#         np.mean(masked_a), np.std(masked_a),
#         np.mean(masked_b), np.std(masked_b)
#     ]
#     return np.array(features)

# def extract_features_from_image_obj(image):
#     # 1. 分割
#     masked_image, mask = rgb_bgr_to_hsv_mask(image)
    
#     # 2. 提取特征
#     fv_hu_moments = fd_hu_moments(masked_image)
#     fv_haralick   = fd_haralick(masked_image)
#     fv_histogram  = fd_histogram(masked_image, mask)
#     fv_color_stats = fd_color_stats(image, mask)
    
#     # 3. 拼接
#     return np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_color_stats])


# # =================第二步：数据加载 (带增强)=================

# print("开始处理训练集 (使用 SVM + 数据增强)...")

# global_features = []
# labels = []
# train_categories = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']

# for category in train_categories:
#     path = os.path.join(TRAIN_PATH, category)
#     images = glob.glob(os.path.join(path, "*.png")) 
#     print(f"处理类别: {category}, 数量: {len(images)}")
    
#     for image_file in images:
#         image = cv2.imread(image_file)
#         image = resize_image(image)
        
#         # 1. 原图
#         global_features.append(extract_features_from_image_obj(image))
#         labels.append(category)
        
#         # 2. 水平翻转 (增加数据量，防止过拟合)
#         global_features.append(extract_features_from_image_obj(cv2.flip(image, 1)))
#         labels.append(category)
        
#         # 3. 垂直翻转
#         global_features.append(extract_features_from_image_obj(cv2.flip(image, 0)))
#         labels.append(category)

# print(f"特征提取完成。样本数: {len(global_features)}")

# # =================第三步：模型训练 (换回 SVM)=================

# le = LabelEncoder()
# target = le.fit_transform(labels)

# scaler = StandardScaler()
# global_features_scaled = scaler.fit_transform(np.array(global_features))

# print("开始训练 SVM 模型...")

# # 使用 SVM，并在参数网格中寻找最优解
# # RBF核通常是处理这类非线性特征最好的选择
# param_grid = [
#     {
#         'C': [1, 10, 100, 1000],         # 惩罚系数，越大越容易过拟合，越小越欠拟合
#         'gamma': [1, 0.1, 0.01, 0.001],  # 核函数系数
#         'kernel': ['rbf']
#     },
#     # 也可以尝试线性核，有时候简单更有效
#     # {'C': [1, 10, 100], 'kernel': ['linear']} 
# ]

# # 使用 StratifiedKFold 保证验证集类别比例一致
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# model = GridSearchCV(
#     SVC(probability=True, random_state=42), 
#     param_grid, 
#     refit=True, 
#     verbose=1, 
#     cv=kfold,
#     scoring='accuracy' # 这里直接看准确率
# )

# model.fit(global_features_scaled, target)

# print(f"最佳参数: {model.best_params_}")
# print(f"最佳验证集准确率: {model.best_score_}") 

# # =================第四步：预测与保存=================

# print("预测测试集...")
# sample_submission = pd.read_csv(SUBMISSION_TEMPLATE)
# test_ids = sample_submission['ID'].values

# test_features = []
# valid_ids = []

# for filename in test_ids:
#     img_path = os.path.join(TEST_PATH, filename)
#     if os.path.exists(img_path):
#         image = cv2.imread(img_path)
#         image = resize_image(image) # 保持一致
#         feature = extract_features_from_image_obj(image)
#         test_features.append(feature)
#         valid_ids.append(filename)

# test_features_scaled = scaler.transform(np.array(test_features))
# predictions = model.predict(test_features_scaled)
# predicted_labels = le.inverse_transform(predictions)

# submission_df = pd.DataFrame({'ID': valid_ids, 'Category': predicted_labels})
# submission_df.to_csv("submission_svm_optimized.csv", index=False)
# print("已保存 submission_svm_optimized.csv")



import os
import glob
import cv2
import numpy as np
import pandas as pd
import mahotas
from skimage.feature import local_binary_pattern # 新增：需要这个库
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# =================配置路径=================
TRAIN_PATH = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\train"
TEST_PATH = r"C:\Users\12132\Desktop\ML1\dataset-for-task1\dataset-for-task1\test"    
SUBMISSION_TEMPLATE = r"C:\Users\12132\Desktop\ML1\submission-for-task1.csv"

BINS = 8 

# =================特征提取工具箱=================

def resize_image(image, max_size=(800, 800)):
    h, w = image.shape[:2]
    ratio = min(max_size[0] / h, max_size[1] / w)
    new_dims = (int(w * ratio), int(h * ratio))
    return cv2.resize(image, new_dims)

def apply_clahe(image):
    """
    新增优化：CLAHE 增强对比度
    让叶脉纹理在不同光照下更清晰，极大增强纹理特征的有效性
    """
    # 转换到 LAB 空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对 L 通道 (亮度) 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # 合并回去
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def rgb_bgr_to_hsv_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 保持稍微宽松的绿色范围
    lower_green = np.array([25, 40, 40]) 
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 保持微小的去噪核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    # 对 Hu 矩进行 Log 变换，使其分布更适合 SVM
    feature = -np.sign(feature) * np.log10(np.abs(feature) + 1e-10)
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        haralick = mahotas.features.haralick(gray).mean(axis=0)
    except ValueError:
        return np.zeros(13)
    return haralick

def fd_lbp(image):
    """
    新增特征：Local Binary Patterns (LBP)
    这是专门针对纹理分类的强大特征，弥补 Haralick 的不足
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # LBP 参数
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    
    # 计算 LBP 直方图
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # 归一化
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], mask, [BINS, BINS, BINS], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_color_stats(image, mask=None):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    masked_l = l[mask > 0]
    masked_a = a[mask > 0]
    masked_b = b[mask > 0]
    if len(masked_l) == 0: return np.zeros(6)
    return np.array([np.mean(masked_l), np.std(masked_l), np.mean(masked_a), np.std(masked_a), np.mean(masked_b), np.std(masked_b)])

def extract_features_from_image_obj(image):
    # 0. 预处理：CLAHE 增强
    image = apply_clahe(image)

    # 1. 分割
    masked_image, mask = rgb_bgr_to_hsv_mask(image)
    
    # 2. 提取特征
    fv_hu_moments = fd_hu_moments(masked_image)
    fv_haralick   = fd_haralick(masked_image)
    fv_histogram  = fd_histogram(masked_image, mask)
    fv_color_stats = fd_color_stats(image, mask)
    fv_lbp        = fd_lbp(masked_image) # 加入 LBP
    
    # 3. 拼接 (特征维度增加了)
    return np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_color_stats, fv_lbp])

# =================数据处理=================

print("开始处理训练集 (CLAHE + LBP + Augmentation)...")

global_features = []
labels = []
train_categories = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']

for category in train_categories:
    path = os.path.join(TRAIN_PATH, category)
    images = glob.glob(os.path.join(path, "*.png")) 
    print(f"类别: {category}, 数量: {len(images)}")
    
    for image_file in images:
        image = cv2.imread(image_file)
        image = resize_image(image)
        
        # 1. 原图
        global_features.append(extract_features_from_image_obj(image))
        labels.append(category)
        
        # 2. 水平翻转
        global_features.append(extract_features_from_image_obj(cv2.flip(image, 1)))
        labels.append(category)
        
        # 3. 垂直翻转
        global_features.append(extract_features_from_image_obj(cv2.flip(image, 0)))
        labels.append(category)

print(f"特征提取完成。总样本: {len(global_features)}, 特征维度: {np.array(global_features).shape}")

# =================模型训练 (Voting Ensemble)=================

le = LabelEncoder()
target = le.fit_transform(labels)

scaler = StandardScaler()
global_features_scaled = scaler.fit_transform(np.array(global_features))

print("开始训练集成模型 (SVM + Random Forest)...")

# 1. 定义 SVM 模型 (使用之前 GridSearch 找到的较好参数，这里设为通用强参数)
# C=100, gamma=0.001 是 rbf 核的经典组合
clf_svm = SVC(kernel='rbf', C=100, gamma=0.001, probability=True, random_state=42)

# 2. 定义 随机森林 模型 (增加树的数量)
clf_rf = RandomForestClassifier(n_estimators=300, random_state=42)

# 3. 定义 投票分类器 (Soft Voting 使用概率平均，比硬投票更准)
# 权重 weights=[2, 1] 表示 SVM 的意见比 随机森林 重要两倍（因为 SVM 在此任务通常表现更好）
eclf = VotingClassifier(estimators=[('svm', clf_svm), ('rf', clf_rf)], voting='soft', weights=[1, 1])

# 简单的交叉验证看一下分数
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(eclf, global_features_scaled, target, cv=kfold, scoring='accuracy')
print(f"集成模型 5折交叉验证平均准确率: {scores.mean():.4f}")

# 全量训练
eclf.fit(global_features_scaled, target)

# =================预测与保存=================

print("预测测试集...")
sample_submission = pd.read_csv(SUBMISSION_TEMPLATE)
test_ids = sample_submission['ID'].values

test_features = []
valid_ids = []

for filename in test_ids:
    img_path = os.path.join(TEST_PATH, filename)
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        image = resize_image(image)
        feature = extract_features_from_image_obj(image)
        test_features.append(feature)
        valid_ids.append(filename)

test_features_scaled = scaler.transform(np.array(test_features))
predictions = eclf.predict(test_features_scaled)
predicted_labels = le.inverse_transform(predictions)

submission_df = pd.DataFrame({'ID': valid_ids, 'Category': predicted_labels})
submission_df.to_csv("12_04_submission_ensemble_clahe_lbp.csv", index=False)
print("结果已保存至 submission_ensemble_clahe_lbp.csv")
