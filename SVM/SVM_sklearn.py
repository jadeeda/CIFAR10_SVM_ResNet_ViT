import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
import torchvision
from skimage.feature import hog
from tqdm import tqdm
import joblib  # 新增导入

# 1. 数据加载与预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 提取训练集和测试集（内存允许建议使用全部数据）
X_train, y_train = trainset.data[:50000], np.array(trainset.targets)[:50000]
X_test, y_test = testset.data, np.array(testset.targets)

# 2. 特征工程（HOG + 颜色直方图）
def extract_features(images):
    hog_features = []
    color_features = []
    for img in tqdm(images):
        # HOG特征（调整参数可优化）
        fd = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), 
                visualize=False, channel_axis=-1)
        hog_features.append(fd)
        
        # 颜色直方图（分通道）
        hist_r = np.histogram(img[:,:,0], bins=32, range=(0,256))[0]
        hist_g = np.histogram(img[:,:,1], bins=32, range=(0,256))[0]
        hist_b = np.histogram(img[:,:,2], bins=32, range=(0,256))[0]
        color_features.append(np.concatenate([hist_r, hist_g, hist_b]))
    
    # 合并特征并标准化
    features = np.hstack([hog_features, color_features])
    return StandardScaler().fit_transform(features)

# 提取特征（耗时较长，建议分批处理）
X_train_feats = extract_features(X_train)
X_test_feats = extract_features(X_test)

# 3. PCA降维（加速训练）
print("3. PCA降维（加速训练）")
pca = PCA(n_components=300, whiten=True)
X_train_pca = pca.fit_transform(X_train_feats)
X_test_pca = pca.transform(X_test_feats)

print("4. 参数调优与模型训练")
# 4. 参数调优与模型训练
param_grid = {
    'C': [0.001],
    'gamma': [0.001],
    'kernel': ['rbf']
}

svm = SVC(cache_size=1000, verbose=0)
grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

cv_results = grid_search.cv_results_
print(cv_results)
# 5. 评估最佳模型
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_pca)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.5f}")
print(f"Best Parameters: {grid_search.best_params_}")

cm = confusion_matrix(y_test, y_pred)
np.savetxt('confusion_matrix.csv', cm, delimiter=',', fmt='%d')

# # 新增模型保存部分
model_filename = 'svm_cifar10_model2.pkl'
joblib.dump(best_svm, model_filename)  # 保存最佳模型
print(f"Model saved to {model_filename}")

# 新增PCA对象保存（确保特征维度一致）
pca_filename = 'pca_model2.pkl'
joblib.dump(pca, pca_filename)
print(f"PCA model saved to {pca_filename}")

# 新增完整模型加载示例（需放在新脚本中）

# 加载已保存的模型和预处理组件
loaded_model = joblib.load(model_filename)
loaded_pca = joblib.load(pca_filename)

print(loaded_model.get_params())
print(loaded_pca.get_params())
# 使用加载的模型进行预测（需确保特征提取方式一致）
def predict_loaded_model(image):
    # 预处理
    features = extract_features([image])
    features_pca = loaded_pca.transform(features)
    return loaded_model.predict(features_pca)

# 测试加载模型

test_sample = X_test[1000]
print(f"Loaded Model Prediction: {predict_loaded_model(test_sample)}")
print(f"Actual Label: {y_test[1000]}")