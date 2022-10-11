from ast import operator
from itertools import count
from multiprocessing.sharedctypes import Value

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Model, load_model
from keras.utils import np_utils
from matplotlib.image import imsave
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tensorflow.keras.applications import xception

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
deep_model = K.function([model.layers[3].output],
                                  [model.output])
layer_output = get_3rd_layer_output([x_train])[0]
print(np.shape(layer_output))
layer_output = np.mean(layer_output,axis=(1, 2))
print(np.shape(layer_output))

clf=DecisionTreeClassifier(max_depth=18, min_samples_leaf=6)
clf.fit(layer_output,predictions)


n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

node_indicator = clf.decision_path(layer_output)
predictions2 = clf.predict(layer_output)

pcount = 0
for i in range(len(predictions)):
    if predictions[i] == predictions2[i]:
        pcount = pcount+1
print(pcount/len(predictions))

def get_integrated_gradients(img_input, top_pred_idx, baseline=None, num_steps=10):
    img_size = (1,8,8,16)

    if baseline is None:
        baseline = np.zeros(img_size)
    else:
        baseline = baseline

    img_input = img_input
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image)

    grads = []
    for i, img in enumerate(interpolated_image):
        # img = tf.expand_dims(img, axis=0)
        grad = get_gradients(img, top_pred_idx=top_pred_idx)
        grads.append(grad)

    avg_grads = np.mean(grads, axis=0)
    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads

def get_gradients(img_input, top_pred_idx):
    # images = tf.convert_to_tensor(img_input, tf.float32)

    # with tf.GradientTape() as tape:
    #     tape.watch(images)
    #     preds = deep_model(images)[0]
    #     top_class = preds[:, top_pred_idx]

    # top_class = tf.py_function(tf.convert_to_tensor(top_class))
    # grads = tape.gradient(top_class, images)
    # print(type(grads))

    grads = K.gradients(model.output[..., top_pred_idx], model.layers[3].output)[0]
    iterate = K.function([model.layers[3].output], grads)
    perturb = iterate([img_input])
    return perturb

def get_path(sam):
    # sam = np.expand_dims(sam,axis=0)
    model_3_layer_out = get_3rd_layer_output([sam])[0]
    model_3_layer_out = np.mean(model_3_layer_out,axis=(1, 2))
    path = clf.decision_path(model_3_layer_out)[0]
    path = path.indices[path.indptr[0]:path.indptr[1]]
    return path

def get_robust(img, path):
    model_3_layer_out = get_3rd_layer_output([img])[0]
    model_3_layer_out = np.mean(model_3_layer_out,axis=(1, 2))
    min = 100
    for i in range(len(path)):
        grads = K.mean(model.layers[3].output[..., feature[path[i]]])
        grads = K.gradients(grads, model.input)[0]
        iterate = K.function([model.input], grads)
        perturb = iterate([img])
        perturb=np.linalg.norm(perturb)
        value = model_3_layer_out[0][feature[path[i]]]
        t = threshold[path[i]]
        min = np.min([min, (t-value)*(t-value)*perturb*perturb])
    return min

# def getFeatureSorted(img, path):
#     model_3_layer_out = get_3rd_layer_output([img])[0]
#     model_3_layer_out = np.mean(model_3_layer_out,axis=(1, 2))
#     map = {}
#     for i in range(len(path)):
#         grads = K.mean(model.layers[3].output[..., feature[path[i]]])
#         grads = K.gradients(grads, model.input)[0]
#         iterate = K.function([model.input], grads)
#         perturb = iterate([img])
#         perturb=np.linalg.norm(perturb)
#         value = model_3_layer_out[0][feature[path[i]]]
#         t = threshold[path[i]]
#         map[feature[path[i]]] = (t-value)*(t-value)*perturb*perturb
#     map = sorted(map.items(), key = lambda kv:kv[1])
#     return map

def init_allpaths(layer_output, node_indicator):
    for i in range(len(layer_output)):
        node_index = node_indicator.indices[node_indicator.indptr[i]:
                                    node_indicator.indptr[i + 1]]
        pos_node = []
        for node in node_index:
            if layer_output[i,feature[node]]>threshold[node]:
                pos_node.append(node)
        init_coverage(node_index, pos_node)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = clf.apply(layer_output)

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

learning_step = 0.01

total_norm = 0

total_perturb_adversial = 0

adversial_num = 0

save_dir = './generated_inputs/features_test/' 

ratio = []

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import time

start_1 = time.time()

for i in range(img_num):

    img_list = []

    img_path = os.path.join(img_dir,img_paths[i])

    img_name = img_paths[i].split('.')[0]

    mannul_label = int(img_name.split('_')[1])

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()

    img_list.append(tmp_img)
    
    path = get_path(tmp_img)

    print(img_path)
    # print(get_robust(tmp_img,path))

    update_coverage(path)

    while len(img_list) >0:

        # count = 0

        gen_img = img_list[0]

        img_list.remove(gen_img)

        pred1 = model.predict(gen_img)

        label1 = np.argmax(pred1[0])

        label_top5 = np.argsort(pred1[0])[-5:]

        path = get_path(gen_img)

        # map = getFeatureSorted(gen_img, path)

        update_coverage(path)

        orig_label = label1
        orig_pred = pred1

        gen_copy = gen_img.copy()

        newPath = path

        for i in range(1):

            gen_img = gen_copy

            img_mid = get_3rd_layer_output([gen_img])[0]

            importance_map = get_integrated_gradients(img_mid, label1)

            loss_neuron = feature_selection(model, 8,8,16, importance_map[0])

            # if path_convert(path)!=path_convert(newPath):
            #     print("find target path")
            #     loss_neuron = path_selection(model, path_convert(path), path_convert(newPath), feature=feature)
            # elif i==1:
            #     break
            # #将上述两个目标合并

            layer_output = -200 * K.sum(loss_neuron)

            newPath = path

            final_loss = K.mean(layer_output)

            grads = K.gradients(final_loss, model.input)[0]

            grads_m = (K.sqrt(K.mean(K.square(grads))) + 1e-5)

            grads = grads / grads_m

            grads_tensor_list = []
            grads_tensor_list.extend(loss_neuron)
            grads_tensor_list.append(grads)

            #得到最后的目标函数
            iterate = K.function([model.input], grads_tensor_list)

            #对于每个样本最多进行5次扰动
            for iter in range(5):
                #获得针对特定样本的具体扰动，得到混淆样本
                loss_neuron_list = iterate([gen_img])

                perturb = loss_neuron_list[-1]*learning_step

                gen_img += perturb
                
                newPath = get_path(gen_img)

                previous_coverage = path_covered(total_path)

                pred1 = model.predict(gen_img)
                label1 = np.argmax(pred1[0])

                current_coverage = path_covered(total_path)

                diff_img = gen_img - orig_img

                L2_norm = np.linalg.norm(diff_img)

                orig_L2_norm = np.linalg.norm(orig_img)

                #计算l2范数，得到扰动程度
                perturb_adversial = L2_norm/orig_L2_norm
                
                # #扰动程度小的话且与原样本路径不同，则加入样本集中继续扰动
                # if perturb_adversial <0.01 and label1==mannul_label:
                #     img_list.append(gen_img)

                #将模型识别错误的样本加入对抗样本集中
                if label1!=orig_label:
                    #更新方法此时已覆盖的路径
                    update_coverage(get_path(gen_img))

                    total_norm += L2_norm

                    total_perturb_adversial += perturb_adversial

                    gen_img_tmp = gen_img.copy()

                    gen_img_deprecessed = deprocess_image(gen_img_tmp)

                    save_img = save_dir + img_name.split('_')[0]+ '_'+ str(mannul_label) + '_'+ str(label1)+ '_' + str(perturb_adversial) + '.png'

                    imsave(save_img, gen_img_deprecessed, cmap='gray')

                    # count = count+1
                    
                    # if i==0:
                    #     break

                    # if count > 3:
                    #     break

    print('covered percentage %.3f'
        % path_covered(total_path)[2])
    ratio.append(path_covered(total_path)[2])

end_1 = time.time()
print('耗时：'+str(end_1-start_1))
import matplotlib.pyplot as plt

x = [ i for i in range(len(ratio)) ]

plt.plot(x, ratio)
plt.show()

print('average_norm = ' + str(total_norm / adversial_num))
print('adversial num = ' + str(adversial_num))
print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))
