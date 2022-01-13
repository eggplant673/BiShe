from keras.backend import reshape
from keras.models import load_model
from keras import models
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def load_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train.reshape(len(x_train),28, 28, 1)
  x_test = x_test.reshape(len(x_test),28 ,28, 1)
  y_train = np_utils.to_categorical(y_train, 10)
  y_test = np_utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data() 
model = load_model('mnist_model2.h5')

import matplotlib.pyplot as plt

from keras import backend as K


from keras.preprocessing import image
import numpy as np

# `img` is a PIL image of size 224x224
x = x_train[729]
plt.imshow(x)
plt.show()
# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)# 增加一个维度，便于后续的计算
predict_class = np.argmax(model.predict(x))

number_output = model.output[:, predict_class]#目标输出估值

last_conv_layer = model.get_layer('cov2')#最后一个卷积层的输出特征图
grads = K.gradients(number_output, last_conv_layer.output)[0]#类别相对于 输出特征图的梯度
print(np.shape(grads))

pooled_grads = K.mean(grads, axis=(0,1,2))# 形状为(512,)的向量，每个元素是特定特征图通道的梯度平均大小
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])# 访问刚刚定义的量，对于给定的样本图像pooled_grads和block_conv3层的输出特征图

pooled_grads_value, conv_layer_output_value = iterate([x])# 图像的numpy数组形式
# 将特征图组的每个通道乘以“重要程度”

import cv2
for i in range(conv_layer_output_value.shape[-1]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]


# size = 28
# margin = 5

# # This a empty (black) image where we will store our results.
# results = np.zeros((4 * size + 7 * margin, 3 * size + 7 * margin))# numpy数组来存放结果，结果与结果之间有黑色的边际

# for i in range(4):  # iterate over the rows of our results grid，行循环
#     for j in range(3):  # iterate over the columns of our results grid，列循环
#         # Generate the pattern for filter `i + (j * 8)` in `layer_name`
#         heatmap = np.maximum(conv_layer_output_value[:, :, i+4*j], 0)
#         heatmap /= np.max(heatmap)+ 1e-5
#         filter_img =cv2.resize(heatmap, (28, 28),interpolation = cv2.INTER_LINEAR)
#         heatmap = np.uint8(255 * heatmap)
#         # Put the result in the square `(i, j)` of the results grid
#         horizontal_start = i * size + i * margin# 水平方向开始的调整
#         horizontal_end = horizontal_start + size# 水平方向结束的调整
#         vertical_start = j * size + j * margin# 垂直开始方向的调整
#         vertical_end = vertical_start + size# 垂直结束方向的调整
#         results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img# 最终所有过滤器的可视化结果

# # Display the results grid

# plt.imshow(results)
# plt.show()


heatmap = np.mean(conv_layer_output_value, axis=-1)#得到的特征图的逐通道平均值就是类激活热力图

# 规范化
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

import cv2
x = np.reshape(x,(28,28))

heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]),interpolation = cv2.INTER_LINEAR)
heatmap = np.uint8(255 * heatmap)

from keras.preprocessing.image import image
# 将热力图应用于原始图像
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)

# # 0.4 为热力图强度因子
# superimposed_img = heatmap * 0.4 + x

plt.imshow(x)
plt.imshow(heatmap,cmap = plt.cm.jet, alpha = 0.3, interpolation='bilinear')
plt.colorbar()
plt.show()
