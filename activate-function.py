import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

# 生成数据
x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# 绘制 Sigmoid 函数
plt.plot(x, y_sigmoid, 'k')
plt.title('')
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([0, 0.5, 1])
plt.savefig('sigmoid.svg')
plt.show()

# 绘制 Tanh 函数
plt.plot(x, y_tanh, 'k')
plt.title('')
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.savefig('tanh.svg')
plt.show()

# 绘制 ReLU 函数
plt.plot(x, y_relu, 'k')
plt.title('')
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([0, 5, 10])
plt.savefig('relu.svg')
plt.show()

# 绘制 Leaky ReLU 函数
plt.plot(x, y_leaky_relu, 'k')
plt.title('')
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([0,2,4,6,8,10])
plt.savefig('leaky_relu.svg')
plt.show()
