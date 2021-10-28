from os import times
from typing import List, Set
from keras.layers import serialization
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from sklearn.model_selection import train_test_split
from sklearn import tree
import joblib

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

def load_mnist():
    
    path = r'D:\BaiduNetdiskDownload\mnist.npz' #放置mnist.py的目录。注意斜杠
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def imgChange(img, pixels, inc):
    for pixel in pixels:
        if img[pixel] < 255-inc:
            img[pixel] += inc

def imgDChange(img, pixels, dec):
    for pixel in pixels:
        if img[pixel] > dec:
            img[pixel] -= dec
 
(x_train,y_train),(x_test,y_test) = load_mnist()
# print(x_test.shape)
# plt.imshow(x_test[0])
# plt.show()

# 训练，预测
import pickle
clf = joblib.load('mnist2.pkl')

x_train = x_train.reshape((60000,784))
x_test = x_test.reshape((10000,784))


# from keras.models import load_model
# model = load_model('my_model.h5')
# clf=DecisionTreeClassifier()
# clf.fit(x_train,np.argmax(model.predict(x_train),axis=1))
# print(clf.score(x_train,y_train))
# print(clf.score(x_test,y_test))

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold


node_indicator = clf.decision_path(x_train)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = clf.apply(x_train)

# HERE IS WHAT YOU WANT
sample_id = 8531
import copy
origin_img = copy.deepcopy(x_train[sample_id]) 
plt.imshow(x_train[sample_id].reshape(28,28),cmap='gray')
plt.show()
# plt.imshow(x_train[2].reshape(28,28))
# plt.show()

node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]


print('Rules used to predict sample %s: %s' % (sample_id,y_train[sample_id]))
pixels = []
for node_id in node_index:

    if leave_id[sample_id] == node_id:  # <-- changed != to ==
        #continue # <-- comment out
        print("leaf node {} reached, no decision here".format(leave_id[sample_id])) # <--

    else: # < -- added else to iterate through decision nodes
        if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 x_train[sample_id, feature[node_id]], # <-- changed i to sample_id
                 threshold_sign,
                 threshold[node_id]))
        pixels.append(feature[node_id])
# joblib.dump(clf,'mnist2.pkl')
# joblib.dump(clf,'mnist.pkl')

    # # 找出每个数字决策树偏好判断的像素点
    # indexLength = 0
    # indexfor9 = []
    # count9 = [0]*784

    # # 使用的像素点统计
    # sample_ids_for_9 = [i for i in range(len(x_train)) if clf.predict([x_train[i]])==[9]]
    # for id in sample_ids_for_9:
    #     node_index = node_indicator.indices[node_indicator.indptr[id]:
    #                                     node_indicator.indptr[id + 1]]
    #     for node_id in node_index:             
    #         count9[feature[node_id]]+=1

    # # 确定最少需要的像素点
    # indexfor9 = [i for i in range(784) if count9[i]>=len(sample_ids_for_9)*(0.4)]
    # print(indexfor9)

indexLength = 0
index = []
times = 0
count = [0]*784
# 使用的像素点统计
sample_ids = [i for i in range(len(x_train)) if clf.predict([x_train[i]])==[y_train[sample_id]]]
for id in sample_ids:
    node_index = node_indicator.indices[node_indicator.indptr[id]:
                                    node_indicator.indptr[id + 1]]
    for node_id in node_index:             
        count[feature[node_id]]+=1

# 确定最少需要的像素点
while indexLength<20:
    index = [i for i in range(784) if count[i]>=len(sample_ids)*(0.8-0.1*times)]
    indexLength = len(index)
    times+=0.5

img = [0]*784
for i in index:
    img[i]=255
print(index)
plt.imshow(np.array(img).reshape(28,28),cmap='gray')
plt.show()

from keras.models import load_model
model = load_model('mnist_model2.h5')
aa=model.predict(x_train.reshape(len(x_train),28, 28, 1))[sample_id]
print(aa)

# 改变决策分支所依赖的像素点的值
newIndex = []
incIndex = []
decIndex = []
BackgroudFuzz = 10
originResult = np.argmax(model.predict(x_train.reshape(len(x_train),28, 28, 1))[sample_id])
mutatedResult = originResult

indexNotIn = list(set(index)-set(pixels))
decIndex = []
print(indexNotIn)

def addToIndex(toindex, pixel):
    toindex.append(pixel)
    toindex.append(pixel+1)
    toindex.append(pixel-1)
    toindex.append(pixel-28)
    toindex.append(pixel-29)
    toindex.append(pixel-27)
    toindex.append(pixel+28)
    toindex.append(pixel+29)
    toindex.append(pixel+27) 
    # 周围两格内
    toindex.append(pixel+56)
    toindex.append(pixel+55)
    toindex.append(pixel+57)
    toindex.append(pixel-56)
    toindex.append(pixel-55)
    toindex.append(pixel-57)
    toindex.append(pixel+2)
    toindex.append(pixel-2)
    toindex.append(pixel-58)
    toindex.append(pixel-54)
    toindex.append(pixel-30)
    toindex.append(pixel-26)
    toindex.append(pixel+26)
    toindex.append(pixel+30)
    toindex.append(pixel+54)
    toindex.append(pixel+58)
    
# for pixel in indexNotIn:
#         if(pixel>60 and pixel < 720 ):
#             addToIndex(newIndex,pixel)
for node_id in node_index:
    if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
        # x_train[sample_id, feature[node_id]] = threshold[node_id]+(255-threshold[node_id])*changeRatio
        if(feature[node_id] > 60 and feature[node_id] < 720 ):
            addToIndex(newIndex,feature[node_id])
    else:
        if(feature[node_id] > 60 and feature[node_id] < 720 ):
            addToIndex(decIndex,feature[node_id]) 

while(mutatedResult==originResult and BackgroudFuzz<100):   
    imgDChange(x_train[sample_id], list(set(decIndex)), 10)
    imgChange(x_train[sample_id],list(set(newIndex)),10)
    mutatedResult = np.argmax(model.predict(x_train.reshape(len(x_train),28, 28, 1))[sample_id])        
    BackgroudFuzz += 10

print(model.predict(x_train.reshape(len(x_train),28, 28, 1))[sample_id])
print(mutatedResult)
diff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
print(diff)
plt.imshow(x_train[sample_id].reshape(28,28),cmap='gray')
plt.show()