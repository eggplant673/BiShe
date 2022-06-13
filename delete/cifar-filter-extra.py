from keras.backend import reshape
from keras.datasets import cifar10
from keras.models import load_model
from keras import models
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
x_img_train = x_img_train.astype('float32') / 255.0
x_img_test = x_img_test.astype('float32') / 255.0


model = load_model('cifar_10_model.h5')

import matplotlib.pyplot as plt
from keras import backend as K


from keras.preprocessing import image
import numpy as np

# `img` is a PIL image of size 224x224
x = x_img_train[8]
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

pool_mean = np.mean(conv_layer_output_value,axis=(0, 1))
filter_number = np.argmax(pool_mean)

heatmap = np.mean(conv_layer_output_value, axis=-1)#得到的特征图的逐通道平均值就是类激活热力图
# heatmap = conv_layer_output_value[:,:,filter_number]
# 规范化
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

import cv2
x = np.reshape(x,(32,32,3))

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
