from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
# 载入数据集
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
x_img_train = x_img_train.astype('float32') / 255.0
x_img_test = x_img_test.astype('float32') / 255.0
x_img_train_tree = np.array(x_img_train).reshape(len(x_img_train),-1)

from keras.models import load_model
from typing import List, Set
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 决策树生成
# clf=DecisionTreeClassifier()
# clf.fit(x_img_train_tree,np.argmax(model.predict(x_img_train),axis=1))
import joblib
clf = joblib.load('cifar10.pkl')

model = load_model('cifar_10_model.h5')
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold
node_indicator = clf.decision_path(x_img_train_tree)

leave_id = clf.apply(x_img_train_tree)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

# HERE IS WHAT YOU WANT
sample_id = 9
plt.imshow(x_img_train[sample_id])
plt.show()
# plt.imshow(x_img_train[sample_id])
# plt.show()
# plt.imshow(x_train[2].reshape(28,28))
# plt.show()

node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

print('Rules used to predict sample %s: %s' % (sample_id,y_label_train[sample_id]))
print(model.predict(x_img_train)[sample_id])
result = np.argmax(model.predict(x_img_train)[sample_id])
for node_id in node_index:

    if leave_id[sample_id] == node_id:  # <-- changed != to ==
        #continue # <-- comment out
        print("leaf node {} reached, no decision here".format(leave_id[sample_id])) # <--

    else: # < -- added else to iterate through decision nodes
        if (x_img_train_tree[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 x_img_train_tree[sample_id, feature[node_id]], # <-- changed i to sample_id
                 threshold_sign,
                 threshold[node_id]))
# import joblib
# joblib.dump(clf,'cifar10.pkl')

# 找出每个数字决策树偏好判断的像素点
modelResult = model.predict([x_img_train])
sample_ids = [i for i in range(len(x_img_train_tree)) if  np.argmax(modelResult[i]) == result]
print(len(sample_ids))
pixelNum = 32*32*3
count = [0]*pixelNum 
for id in sample_ids:
    node_index = node_indicator.indices[node_indicator.indptr[id]:
                                    node_indicator.indptr[id + 1]]
    for node_id in node_index:             
        count[feature[node_id]]+=1
index = [i for i in range(pixelNum) if count[i]>=len(sample_ids)*0.1]
img = [0]*pixelNum 
for i in index:
    img[i]=255
print(index)
# plt.imshow(np.array(img).reshape(32,32,3))
# plt.show()

attack_sample_ids = [i for i in range(len(x_img_train_tree)) if  np.argmax(modelResult[i]) == 7]
print(len(attack_sample_ids))
pixelNum = 32*32*3
count = [0]*pixelNum 
for id in attack_sample_ids:
    node_index = node_indicator.indices[node_indicator.indptr[id]:
                                    node_indicator.indptr[id + 1]]
    for node_id in node_index:             
        count[feature[node_id]]+=1
index2 = [i for i in range(pixelNum) if count[i]>=len(attack_sample_ids)*0.10]
print(set(index2)-set(index))


def imgChange(img, pixels, changeTo):
    for pixel in pixels:
        if img[pixel] < 1-changeTo:
            img[pixel] += changeTo

def imgDChange(img, pixels, dec):
    for pixel in pixels:
        if img[pixel] > dec:
            img[pixel] -= dec

def addToIndex(index, pixel):
    index.append(pixel)
    index.append(pixel-1*3)
    index.append(pixel-33*3)
    index.append(pixel-31*3)
    index.append(pixel-32*3)
    index.append(pixel+1*3)
    index.append(pixel+31*3)
    index.append(pixel+32*3)
    index.append(pixel+33*3)

Pixels = []
indexUp = []
indexDown = []
#决策单个图片依赖的像素点
for node_id in node_index:
    Pixels.append(feature[node_id])
    if (x_img_train_tree[sample_id, feature[node_id]] <= threshold[node_id]) :
        if feature[node_id] >33*3 and feature[node_id]<pixelNum-33*3:
           addToIndex(indexUp,feature[node_id])
    else:
        if feature[node_id] >33*3 and feature[node_id]<pixelNum-33*3:
           addToIndex(indexDown,feature[node_id])

# indexNotIn = list(set(index2))
# print(indexNotIn)
# for pixel in indexNotIn:
#     if(pixel>33*3 and node_id < 32*32*3 -33*3 ):
#         addToIndex(indexUp,pixel)
 
imgChange(x_img_train_tree[sample_id],indexUp,0.3)
imgDChange(x_img_train_tree[sample_id],indexDown,0.3)

# index = list(set(index2)-set(index))
# for i in index:
#     changeRatio = 2
#     if i >33*3 and i<pixelNum-33*3 and i not in Pixels:
#         x_img_train_tree[sample_id, i] *= changeRatio
#         x_img_train_tree[sample_id, i-1*3] *= changeRatio
#         x_img_train_tree[sample_id, i+1*3] *= changeRatio
#         x_img_train_tree[sample_id, i-32*3] *= changeRatio
#         x_img_train_tree[sample_id, i-33*3] *= changeRatio
#         x_img_train_tree[sample_id, i-31*3] *= changeRatio
#         x_img_train_tree[sample_id, i+32*3] *= changeRatio
#         x_img_train_tree[sample_id, i+31*3] *= changeRatio
#         x_img_train_tree[sample_id, i+33*3] *= changeRatio

plt.imshow(np.array(x_img_train_tree).reshape(len(x_img_train),32,32,3)[sample_id])
print(model.predict(np.array(x_img_train_tree).reshape(len(x_img_train),32,32,3))[sample_id])
print(clf.predict(x_img_train_tree)[sample_id])
plt.show()