import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# ==================== 数据加载与预处理 ====================
def unpickle(file):
    """加载CIFAR-10二进制文件（参考网页6/7/8实现）"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir):
    """加载并合并训练集（5个batch）"""
    train_data, train_labels = [], []
    for i in range(1,6):
        batch = unpickle(f'{data_dir}/data_batch_{i}')
        train_data.append(batch[b'data'])
        train_labels.append(batch[b'labels'])
    
    test_batch = unpickle(f'{data_dir}/test_batch')
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    X_train = np.concatenate(train_data)
    y_train = np.concatenate(train_labels)
    
    # 归一化到[0,1]并转换为float32（参考网页1/4）
    return X_train.astype('float32')/255, y_train, X_test.astype('float32')/255, y_test

# 加载数据集（需替换为实际路径）
X_train, y_train, X_test, y_test = load_cifar10('./data/cifar-10-batches-py')

# 为加速调试，可采样部分数据（参考网页3/5）
X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=10000, stratify=y_train)
X_test, y_test = X_test[:2000], y_test[:2000]

# ==================== SVM模型实现 ====================
class LinearSVM:
    def __init__(self, C=0.1, learning_rate=1e-4, max_iters=5000):
        self.C = C              # 正则化系数
        self.lr = learning_rate # 学习率
        self.max_iters = max_iters
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.random.randn(10, n_features) * 0.01  # 10类权重矩阵
        
        for iter in tqdm(range(self.max_iters)):
            # 计算分类得分矩阵（n_samples x 10）
            scores = X.dot(self.W.T)
            
            # 计算多类Hinge Loss（参考网页4公式）
            correct_scores = scores[np.arange(n_samples), y].reshape(-1,1)
            margins = np.maximum(0, scores - correct_scores + 1)
            margins[np.arange(n_samples), y] = 0  # 忽略正确类别
            
            # 计算损失与梯度
            data_loss = np.sum(margins) / n_samples
            reg_loss = 0.5 * self.C * np.sum(self.W**2)
            total_loss = data_loss + reg_loss
            
            # 梯度计算（矩阵运算优化）
            valid_margin = (margins > 0).astype(int)
            valid_margin[np.arange(n_samples), y] = -np.sum(valid_margin, axis=1)
            dW = (valid_margin.T.dot(X) / n_samples) + self.C * self.W
            
            # 参数更新
            self.W -= self.lr * dW
            
            # 每1轮打印进度
            if iter % 1 == 0:
                y_pred = svm.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
            # print(f"Iter {iter:4d} | Loss: {total_loss:.5f},Accuracy: {accuracy:.5f}")

            with open("SVM-log.txt","a+") as ftrain:
                log="%d,%.5f,%.5f\n" % (iter,total_loss,accuracy*100)
                ftrain.write(log)

    def predict(self, X):
        return np.argmax(X.dot(self.W.T), axis=1)

# ==================== 训练与评估 ====================
if __name__ == "__main__":
    # 初始化模型（参数参考网页5优化建议）
    svm = LinearSVM(C=0.1, learning_rate=1e-4, max_iters=50000)
    
    print("开始训练...")
    svm.train(X_train, y_train)
    
    print("\n评估测试集:")
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"测试准确率: {accuracy*100:.2f}%")