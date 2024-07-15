import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize


# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 二元交叉熵损失函数
def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# 损失函数的导数
def d_binary_crossentropy(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def get_accuracy(x, y, y_pred):
    y_hat = (y_pred > 0.5) * 1
    minus = y - y_hat
    same_label = np.where(minus == 0)
    return len(same_label[0]) / len(y) * 100

# 初始化权重和偏置
input_size = 780
hidden_size = 100
output_size = 1

# 权重矩阵和偏置向量
W1 = 2 * np.random.random((input_size, hidden_size)) - 1
b1 = np.zeros((1, hidden_size))
W2 = 2 * np.random.random((hidden_size, output_size)) - 1
b2 = np.zeros((1, output_size))


# 前向传播
def forward_propagation(X, y):
    global z1, a1, z2
    # 隐藏层的输入和输出
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)

    # 计算损失
    loss = binary_crossentropy(y, output)

    return loss, output


# 反向传播
def backward_propagation(X, y, output):
    # 损失函数对输出的导数，形状应为 (600, 1)
    d_loss_d_output = (output - y)

    # 隐藏层的激活函数的导数，形状为 (600, 1)
    d_output_d_z2 = sigmoid_derivative(output)

    # 损失函数对 W2 的梯度，通过将 d_loss_d_output 与 d_output_d_z2 相乘并乘以 a1 的转置
    d_W2 = np.dot(a1.T, d_loss_d_output * d_output_d_z2)

    # 损失函数对 b2 的梯度
    d_b2 = np.sum(d_loss_d_output * d_output_d_z2, axis=0, keepdims=True)

    # 反向传播到隐藏层
    # 损失函数对 z1 的梯度，通过将 d_loss_d_output 与 W2 转置相乘得到
    d_z1 = np.dot(d_loss_d_output, W2.T)

    # 损失函数对 W1 的梯度
    d_W1 = np.dot(X.T, sigmoid_derivative(a1) * d_z1)

    # 损失函数对 b1 的梯度
    d_b1 = np.sum(sigmoid_derivative(a1) * d_z1, axis=0, keepdims=True)

    return d_W1, d_b1, d_W2, d_b2


# 更新权重和偏置
def update_weights(d_W1, d_b1, d_W2, d_b2, learning_rate=0.01):
    global W1, b1, W2, b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2


# 训练函数
def train(X, y, epochs=2000):
    for epoch in range(epochs):
        loss, output = forward_propagation(X, y)
        acc = get_accuracy(X, y, output)
        d_W1, d_b1, d_W2, d_b2 = backward_propagation(X, y, output)
        update_weights(d_W1, d_b1, d_W2, d_b2)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss}, Acc: {acc}%")


# 假设我们有一些数据
mnist_data = sio.loadmat('mnist.mat')
x = mnist_data['Z']
y = mnist_data['y']
y = (y.astype(int) >= 5) * 1  # 将数字>=5样本设为正例，其他数字设为负例
x = normalize(x, axis=1, norm='l2')

x = x[0: 600]
y = y[0: 600]
# 训练神经网络
train(x, y)