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

def addToIndex2(toindex, toindex2, pixel):
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
    toindex2.append(pixel+56)
    toindex2.append(pixel+55)
    toindex2.append(pixel+57)
    toindex2.append(pixel-56)
    toindex2.append(pixel-55)
    toindex2.append(pixel-57)
    toindex2.append(pixel+2)
    toindex2.append(pixel-2)
    toindex2.append(pixel-58)
    toindex2.append(pixel-54)
    toindex2.append(pixel-30)
    toindex2.append(pixel-26)
    toindex2.append(pixel+26)
    toindex2.append(pixel+30)
    toindex2.append(pixel+54)
    toindex2.append(pixel+58)

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
 
(x_train,y_train),(x_test,y_test) = load_mnist()

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
sample_id = 4998
import copy
origin_img = copy.deepcopy(x_train[sample_id]) 
# plt.imshow(x_train[sample_id].reshape(28,28),cmap='gray')
# plt.show()
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
index1 = []
index2 = []
times = 0
count = [0]*784

# 使用的像素点统计
sample_ids = [i for i in range(len(x_train)) if clf.predict([x_train[i]])==[y_train[sample_id]]]
for id in sample_ids:
    node_index = node_indicator.indices[node_indicator.indptr[id]:
                                    node_indicator.indptr[id + 1]]
    for node_id in node_index:     
        if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):        
            count[feature[node_id]]+=1
        else:
            count[feature[node_id]]-=1


from keras.models import load_model
model = load_model('mnist_model2.h5')
aa=model.predict(np.reshape(x_train[sample_id],(1,28,28,1)))
print(aa)
print(np.argmax(aa))

# 改变决策分支所依赖的像素点的值
newIndex = []
incIndex = []
incIndex2 = []
decIndex = []
decIndex2 = []
BackgroudFuzz = 10
originResult = np.argmax(model.predict(x_train.reshape(len(x_train),28, 28, 1))[sample_id])
mutatedResult = originResult
    
# 确定最少需要的像素点
while indexLength<8:
    index1 = [i for i in range(784) if count[i]>=len(sample_ids)*(0.8-0.1*times)]
    indexLength = len(index1)
    times+=0.5
index2 = [i for i in range(784) if count[i]<=-len(sample_ids)*(0.6)]

print(np.shape(index1))
print(np.shape(index2))

for ele in index1:
    if ele >30 and ele <750:
        addToIndex(incIndex2,ele)
for ele in index2:
    if ele >30 and ele <750:
        addToIndex(decIndex2,ele)

for node_id in node_index:
    if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
        # x_train[sample_id, feature[node_id]] = threshold[node_id]+(255-threshold[node_id])*changeRatio
        if(feature[node_id] > 60 and feature[node_id] < 720 ):
            addToIndex2(incIndex, incIndex2, feature[node_id])
    else:
        if(feature[node_id] > 60 and feature[node_id] < 720 ):
            addToIndex(decIndex, feature[node_id]) 

# 判断改变的像素点是否在决策区域内
# from imageio import imread
# import numpy as np 
# image1 = imread('./seeds_50/729_4.png')
# image2 = imread('./generated_inputs/0602/729_4_94519091363.png')

# print(np.linalg.norm(image1/255-image2/255,ord=2))
# print(model.predict(np.reshape(image2,(1,28,28,1))))
# allIndex = list(set(incIndex + decIndex))

# diff = image2 - image1
# diff = np.reshape(diff,784)
# diff = [ i for i in range(784) if diff[i] != 0]
# incIndex = list(set(incIndex)&set(diff))
# incIndex2 = list(set(incIndex2)|(set(incIndex)-set(diff)))
# decIndex = list(set(decIndex)&set(diff))
# decIndex2 = list(set(decIndex2)|(set(decIndex)-set(diff)))

decIndex2 = list(set(decIndex2)-set(decIndex)-set(incIndex)-set(incIndex2))
decIndex = list(set(decIndex)-set(decIndex2)-set(incIndex)-set(incIndex2))
incIndex2 = list(set(incIndex2)-set(incIndex)-set(decIndex)-set(decIndex2))
incIndex = list(set(incIndex)-set(incIndex2)-set(decIndex)-set(decIndex2))

decIndex = list(set(decIndex)-set(incIndex))

imgDChange(x_train[sample_id], list(set(decIndex2)), 20)
imgChange(x_train[sample_id],list(set(incIndex2)), 20)


while(mutatedResult==originResult and BackgroudFuzz<150):   
    imgDChange(x_train[sample_id], decIndex, 5)
    imgChange(x_train[sample_id], incIndex, 5)
    imgChange(x_train[sample_id], incIndex2, 4)
    mutatedResult = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))      
    BackgroudFuzz += 5

# node_indicator = clf.decision_path(x_train)
# node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                     node_indicator.indptr[sample_id + 1]]

print(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))
print(mutatedResult)
print(BackgroudFuzz)
diff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
print(diff)
plt.imshow(x_train[sample_id].reshape(28,28),cmap='gray')
plt.show()