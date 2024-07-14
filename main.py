import numpy as np
from sklearn.preprocessing import normalize
import scipy.io as sio

# 激活函数及其导数
def sigmoid(x):
    print(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 损失函数
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 初始化参数
input_size = 780
hidden_size = 1000
output_size = 1

np.random.seed(42)
weights_input_to_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.random.randn(hidden_size)
weights_hidden_to_output = np.random.randn(hidden_size, output_size)
bias_output = np.random.randn(output_size)

mnist_data = sio.loadmat('mnist.mat')
x = mnist_data['Z']
y = mnist_data['y']
y = (y.astype(int) >= 5) * 1  # 将数字>=5样本设为正例，其他数字设为负例
# 添加一列全为1的偏置项列
# x = np.hstack((x, np.ones((x.shape[0]))))
# 归一化特征向量
# X = normalize(x, axis=1, norm='l2')
x = normalize(x, axis=1, norm='l2')

# 学习率
learning_rate = 0.01

# 假设有一些训练数据
X_train = np.array(x)
Y_train = np.array(y)

# 训练网络
for epoch in range(1000):  # 迭代1000次
    # 前向传播
    hidden_layer_input = np.dot(x, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    y_pred = sigmoid(output_layer_input)
    # print(y, y_pred)
    # 损失函数
    loss = cross_entropy_loss(y, y_pred)
    # 后向传播
    output_error = -(y - y_pred)
    hidden_error = output_error.dot(weights_hidden_to_output.T) * sigmoid_derivative(hidden_layer_output)

    # 梯度
    output_delta = [o * output_error for o in output_layer_input][0][0]
    hidden_delta = [h * hidden_error for h in hidden_layer_output][0][0]

    # 参数更新
    weights_hidden_to_output -= learning_rate * np.outer(hidden_layer_output, output_delta)
    weights_input_to_hidden -= learning_rate * np.outer(x, hidden_delta)

    # if (epoch + 1) % 100 == 0:
    print(f"Epoch {epoch + 1}, Loss: {loss}")

# 测试网络
print("预测结果:")
for x in X_train:
    hidden = sigmoid(np.dot(x, weights_input_to_hidden)) + bias_hidden
    output = sigmoid(np.dot(hidden, weights_hidden_to_output)) + bias_output
    print(output)