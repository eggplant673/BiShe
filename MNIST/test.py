from matplotlib.image import imsave
import numpy as np
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.layers import Input
from numpy.core.defchararray import center
from numpy.core.fromnumeric import shape, sort
from numpy.lib.function_base import delete
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import tensorflow as tf
from utils import *
tf.compat.v1.disable_eager_execution()

def get_mnist_data():
    (x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()

    # 从训练集中分配验证集
    x_val = x_train_original[50000:]
    y_val = y_train_original[50000:]
    x_train = x_train_original[:50000]
    y_train = y_train_original[:50000]

    # 将图像转换为四维矩阵(nums,rows,cols,channels), 这里把数据从unint类型转化为float32类型, 提高训练精度。
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test_original.reshape(x_test_original.shape[0], 28, 28, 1).astype('float32')

    # 原始图像的像素灰度值为0-255，为了提高模型的训练精度，通常将数值归一化映射到0-1。
    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255

    # 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_test = np_utils.to_categorical(y_test_original)

    return x_train, y_train, x_val, y_val, x_test, y_test

# 加载模型
model = load_model('./MNIST/lenet5.h5')
x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()

# 训练集结果预测
predictions = model.predict(x_train)
predictions = np.argmax(predictions, axis=1)

from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x_train])[0]
layer_output = pool_mean = np.mean(layer_output,axis=(1, 2))
print(np.shape(layer_output))

clf=DecisionTreeClassifier(min_samples_leaf=4)
clf.fit(layer_output,predictions)

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold


node_indicator = clf.decision_path(layer_output)

def get_path(sam):
    # sam = np.expand_dims(sam,axis=0)
    model_3_layer_out = get_3rd_layer_output([sam])[0]
    model_3_layer_out = np.mean(model_3_layer_out,axis=(1, 2))
    path = clf.decision_path(model_3_layer_out)[0]
    path = path.indices[path.indptr[0]:path.indptr[1]]
    return path

def init_allpaths(layer_output, node_indicator):
    for i in range(len(layer_output)):
        node_index = node_indicator.indices[node_indicator.indptr[i]:
                                    node_indicator.indptr[i + 1]]
        init_coverage(node_index)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = clf.apply(layer_output)

# HERE IS WHAT YOU WANT
# sample_id = 1657
leaves = []
for i in range(len(leave_id)):
    leaves.append(leave_id[i])
total_path = len(set(leaves))
print(total_path)

init_allpaths(layer_output, node_indicator)
sort_allpaths()

import os

img_dir = './generated_inputs/new/'
img_paths = os.listdir(img_dir)
img_num = len(img_paths)

# input image dimensions
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

predict_weight = 0.5

learning_step = 0.02

total_norm = 0

total_perturb_adversial = 0

adversial_num = 0

save_dir = './generated_inputs/new/' 

ratio = []

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(img_num):

    img_list = []

    img_path = os.path.join(img_dir,img_paths[i])

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()

    img_list.append(tmp_img)
    
    path = get_path(tmp_img)

    update_coverage(path)

    

    print('covered percentage %.3f'
        % path_covered(total_path)[2])
    ratio.append(path_covered(total_path)[2])

import matplotlib.pyplot as plt


x = [ i for i in range(len(ratio)) ]

plt.plot(x, ratio)
plt.show()
