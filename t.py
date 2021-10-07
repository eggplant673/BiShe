from keras.backend import reshape
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(10)
# # 载入数据集
# (x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
# x_img_train = x_img_train.astype('float32') / 255.0
# x_img_test = x_img_test.astype('float32') / 255.0
# print(np.array(x_img_train).shape)
# from keras.models import load_model
# model = load_model('cifar_10_model.h5')
# aa=model.predict(x_img_train)[66]
# print(aa)
a = [0]*32*32*3
a[4] = 0.7
a[4+33]=0.7
a[4+32*3]=0.7
plt.imshow(np.array(a).reshape(32,32,3))
plt.show()