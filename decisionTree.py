from typing import List, Set
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
 
(x_train,y_train),(x_test,y_test) = load_mnist()
# print(x_test.shape)
# plt.imshow(x_test[0])
# plt.show()

# 训练，预测
import pickle
clf = joblib.load('mnist.pkl')

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

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

# HERE IS WHAT YOU WANT
sample_id = 6
plt.imshow(x_train[sample_id].reshape(28,28))
plt.show()
# plt.imshow(x_train[2].reshape(28,28))
# plt.show()

node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

# 改变决策分支所依赖的像素点的值
# for node_id in node_index:
#     if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
#         x_train[sample_id, feature[node_id]]=threshold[node_id]+1
#     else:
#         x_train[sample_id, feature[node_id]]=threshold[node_id]-1

print('Rules used to predict sample %s: %s' % (sample_id,y_train[sample_id]))
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
# joblib.dump(clf,'mnist.pkl')

# 找出每个数字决策树偏好判断的像素点
sample_ids = [i for i in range(len(x_train)) if clf.predict([x_train[i]])==[y_train[sample_id]]]
print(len(sample_ids))
count = [0]*784
for id in sample_ids:
    node_index = node_indicator.indices[node_indicator.indptr[id]:
                                    node_indicator.indptr[id + 1]]
    for node_id in node_index:             
        count[feature[node_id]]+=1
index = [i for i in range(784) if count[i]>=len(sample_ids)*0.50]
img = [0]*784
for i in index:
    img[i]=255
print(index)
plt.imshow(np.array(img).reshape(28,28))
plt.show()

from keras.models import load_model
model = load_model('my_model.h5')
aa=model.predict(x_train)[sample_id]
print(aa)

# 改变决策分支所依赖的像素点的值
newIndex = []
pixels = []
for pixel in range(784):
    if x_train[sample_id,pixel]!=0:
        pixels.append(pixel)
for node_id in index:
    if(node_id>30 and node_id < 755 ):
        newIndex.append(node_id+1)
        newIndex.append(node_id-1)
        newIndex.append(node_id-28)
        newIndex.append(node_id-29)
        newIndex.append(node_id-27)
        newIndex.append(node_id+28)
        newIndex.append(node_id+29)
        newIndex.append(node_id+27)     
for node_id in index:
    if (x_train[sample_id, node_id] <= threshold[node_id]):
        x_train[sample_id, node_id] = threshold[node_id]+1
    else:
        x_train[sample_id, node_id]= 30
newIndex = list(set(newIndex))
for node_id in newIndex:
        x_train[sample_id, node_id] = 30
plt.imshow(x_train[sample_id].reshape(28,28))
plt.show()

aa=model.predict(x_train)[sample_id]
print(clf.predict([x_train[sample_id]]))
print(aa)