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

clf=DecisionTreeClassifier(min_samples_leaf=2)
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

img_dir = './seeds_50'
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

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(img_num):

    img_list = []

    img_path = os.path.join(img_dir,img_paths[i])

    img_name = img_paths[i].split('.')[0]

    mannul_label = int(img_name.split('_')[1])

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()

    img_list.append(tmp_img)
    
    path = get_path(tmp_img)

    update_coverage(path)

    while len(img_list) >0:

        gen_img = img_list[0]

        img_list.remove(gen_img)

        pred1 = model.predict(gen_img)

        label1 = np.argmax(pred1[0])

        label_top5 = np.argsort(pred1[0])[-5:]

        path = get_path(gen_img)

        update_coverage(path)

        orig_label = label1
        orig_pred = pred1

        loss_1 = K.mean(model.layers[8].output[..., orig_label])
        loss_2 = K.mean(model.layers[8].output[..., label_top5[-2]])
        loss_3 = K.mean(model.layers[8].output[..., label_top5[-3]])
        loss_4 = K.mean(model.layers[8].output[..., label_top5[-4]])
        loss_5 = K.mean(model.layers[8].output[..., label_top5[-5]])
        
        layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

        loss_neuron = path_selection(model,get_path(gen_img), feature, gen_img)

        layer_output += 2 * K.sum(loss_neuron)

        final_loss = K.mean(layer_output)
    
        grads = K.gradients(final_loss, model.input)[0]

        grads_m = (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        grads = grads / grads_m

        grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)

        iterate = K.function([model.input], grads_tensor_list)

        for iter in range(5):

            loss_neuron_list = iterate([gen_img])

            perturb = loss_neuron_list[-1]*learning_step

            gen_img += perturb

            previous_coverage = path_covered(total_path)

            pred1 = model.predict(gen_img)
            label1 = np.argmax(pred1[0])

            update_coverage(get_path(gen_img))

            current_coverage = path_covered(total_path)

            diff_img = gen_img - orig_img

            L2_norm = np.linalg.norm(diff_img)

            orig_L2_norm = np.linalg.norm(orig_img)

            perturb_adversial = L2_norm/orig_L2_norm

            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial <0.1:

                img_list.append(gen_img)

            if label1!=orig_label:

                total_norm += L2_norm

                total_perturb_adversial += perturb_adversial

                gen_img_tmp = gen_img.copy()

                gen_img_deprecessed = deprocess_image(gen_img_tmp)

                save_img = save_dir + img_name + '_'+ str(label1)+ '_' + str(perturb_adversial) + '.png'

                imsave(save_img, gen_img_deprecessed, cmap='gray')

                adversial_num += 1

print('covered neurons percentage:'+ str(path_covered(total_path)[2]))
print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))