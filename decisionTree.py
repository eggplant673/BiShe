from os import times
from typing import List, Set
from keras.layers import serialization
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import center
from numpy.core.fromnumeric import shape, sort
from numpy.lib.function_base import delete
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
def imgs_plus(img1, img2):
    for i in range(len(img1)):
        if img1[i]+img2[i]<=255 and img1[i]+img2[i]>=0:
            img1[i] = img1[i] + img2[i]
        elif img1[i]+img2[i]<0:
            img1[i] = 0
        else:
            img1[i] = 255
        
def imgChange(img, pixels, inc):      
    for pixel in pixels:
        if img[pixel] < 255-inc:
            img[pixel] += inc

def imgDChange(img, pixels, dec):
    for pixel in pixels:
        if img[pixel] > dec:
            img[pixel] -= dec

def dilate_img(img, kernel_size):
    img_size = img.shape[0]
    import copy
    copy_img = copy.deepcopy(img)
    center_move = int(kernel_size/2)
    for i in range(int(kernel_size/2),img_size - int(kernel_size/2)):
        for j in range(int(kernel_size/2),img_size - int(kernel_size/2)):
            img[i, j] = np.max(copy_img[i - center_move:i+center_move, j -center_move:j+center_move])

def erode_img(img, kernel_size):
    img_size = img.shape[0]
    import copy
    copy_img = copy.deepcopy(img)
    center_move = int(kernel_size/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = 0
    for i in range(int(kernel_size/2),img_size - int(kernel_size/2)):
        for j in range(int(kernel_size/2),img_size - int(kernel_size/2)):
            img[i, j] = np.min(copy_img[i - center_move:i+center_move, j -center_move:j+center_move])

def vague_img(img, kernel_size):
    img_size = img.shape[0]
    import copy
    copy_img = copy.deepcopy(img)
    center_move = int(kernel_size/2)
    for i in range(int(kernel_size/2),img_size - int(kernel_size/2)):
        for j in range(int(kernel_size/2),img_size - int(kernel_size/2)):
            img[i, j] = np.mean(copy_img[i - center_move:i+center_move, j -center_move:j+center_move])

def index_process(do_index):
    re_index = []
    tmp = [0]*784
    for i in range(784):
        if i in do_index:
            tmp[i] = 1
    img_tmp = np.reshape(tmp,(28,28))
    erode_img(img_tmp,2)
    tmp = np.reshape(img_tmp,784)
    re_index = [i for i in range(784) if tmp[i]==1]    
    return re_index

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

def imgs_add(img1, img2):
    for i in range(len(img1)):
        if img1[i]+img2[i]<=255:
            img1[i] += img2[i]
        else:
            img1[i] = 255

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
sample_id = 729
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
originResult = np.argmax(model.predict(x_train.reshape(len(x_train),28, 28, 1))[sample_id])
mutatedResult = originResult

# 
# 
# 
# 
# 
# 确定需要的额外的像素点的数量
import copy
origin_img = copy.deepcopy(x_train[sample_id]) 
mindiff = 10
nowdiff = 10
nums = 5
suitablenums = -1
while nums <40 :
    x_train[sample_id] = copy.deepcopy(origin_img)
    # 确定最少需要的像素点
    incIndex = []
    incIndex2 = []
    decIndex = []
    index1 = np.argsort(count)[-1:-nums:-1]
    index2 = np.argsort(count)[0:5:1]

    for node_id in node_index:
        if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
            # x_train[sample_id, feature[node_id]] = threshold[node_id]+(255-threshold[node_id])*changeRatio
            if(feature[node_id] > 60 and feature[node_id] < 720):
                addToIndex2(incIndex, incIndex2, feature[node_id])
                
        else:
            if(feature[node_id] > 60 and feature[node_id] < 720 ):
                addToIndex2(decIndex, decIndex, feature[node_id]) 

    for ele in index1:
        if ele >30 and ele <750:
            addToIndex(incIndex2,ele)

    decIndex = list(set(decIndex)-set(incIndex)-set(incIndex2))
    incIndex2 = list(set(incIndex2)-set(incIndex)-set(decIndex))
    incIndex = list(set(incIndex)-set(incIndex2)-set(decIndex))
    decIndex = list(set(decIndex)-set(incIndex))
    
    # incIndex=index_process(incIndex)
    # incIndex2=index_process(incIndex2)
    BackgroudFuzz = 10
    mutatedResult = originResult
    while(mutatedResult==originResult and BackgroudFuzz<180):   
        imgChange(x_train[sample_id], incIndex, 5)
        imgChange(x_train[sample_id], incIndex2, 5)
        mutatedResult = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))      
        BackgroudFuzz += 5

    # node_indicator = clf.decision_path(x_train)
    # node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
    #                                     node_indicator.indptr[sample_id + 1]]

    nowdiff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
    if mindiff > nowdiff and mutatedResult!=originResult:
        mindiff = nowdiff
        suitablenums =nums
    nums+=2


print(mutatedResult!=originResult)
print(suitablenums)
print(mindiff)


# 
# 
# 
# 
# 
# 确定index2合适的权重
x_train[sample_id] = copy.deepcopy(origin_img)
# 确定最少需要的像素点
incIndex = []
incIndex2 = []
decIndex = []
index1 = np.argsort(count)[-1:-(suitablenums):-1]
index2 = np.argsort(count)[0:5:1]

for node_id in node_index:
    if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
        # x_train[sample_id, feature[node_id]] = threshold[node_id]+(255-threshold[node_id])*changeRatio
        if(feature[node_id] > 60 and feature[node_id] < 720):
            addToIndex2(incIndex, incIndex2, feature[node_id])
            
    else:
        if(feature[node_id] > 60 and feature[node_id] < 720 ):
            addToIndex2(decIndex, decIndex, feature[node_id]) 

for ele in index1:
    if ele >30 and ele <750:
        addToIndex(incIndex2,ele)

decIndex = list(set(decIndex)-set(incIndex)-set(incIndex2))
incIndex2 = list(set(incIndex2)-set(incIndex)-set(decIndex))
incIndex = list(set(incIndex)-set(incIndex2)-set(decIndex))
decIndex = list(set(decIndex)-set(incIndex))
# incIndex=index_process(incIndex)
# incIndex2=index_process(incIndex2)
nowdiff = 10
mindiff = 10
weightfor2 = 0
suitableweightfor2 = 1
while weightfor2 < 10:
    x_train[sample_id] = copy.deepcopy(origin_img)
    BackgroudFuzz = 0
    mutatedResult = originResult
    while(mutatedResult==originResult and BackgroudFuzz<180):   
        imgChange(x_train[sample_id], incIndex, 5)
        imgChange(x_train[sample_id], incIndex2, weightfor2)
        mutatedResult = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))      
        BackgroudFuzz += 5

    # node_indicator = clf.decision_path(x_train)
    # node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
    #                                     node_indicator.indptr[sample_id + 1]]

    nowdiff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
    if nowdiff < mindiff and mutatedResult!=originResult:
        mindiff = nowdiff
        suitableweightfor2 =weightfor2
    weightfor2 += 1

print(suitableweightfor2)
print(mindiff)


x_train[sample_id] = copy.deepcopy(origin_img)
all_pixels = []
# 确定最少需要的混淆程度
incIndex = []
incIndex2 = []
decIndex = []

index1 = np.argsort(count)[-1:-(suitablenums):-1]
index2 = np.argsort(count)[0:5:1]

for node_id in node_index:
    if (x_train[sample_id, feature[node_id]] <= threshold[node_id]):
        # x_train[sample_id, feature[node_id]] = threshold[node_id]+(255-threshold[node_id])*changeRatio
        if(feature[node_id] > 60 and feature[node_id] < 720):
            addToIndex2(incIndex, incIndex2, feature[node_id])
            all_pixels.append(feature[node_id])
            
    else:
        if(feature[node_id] > 60 and feature[node_id] < 720 ):
            addToIndex2(decIndex, decIndex, feature[node_id]) 

for ele in index1:
    if ele >30 and ele <750:
        addToIndex(incIndex2,ele)
        all_pixels.append(ele)


decIndex = list(set(decIndex)-set(incIndex)-set(incIndex2))
incIndex2 = list(set(incIndex2)-set(incIndex)-set(decIndex))
incIndex = list(set(incIndex)-set(incIndex2)-set(decIndex))

decIndex = list(set(decIndex)-set(incIndex))
# incIndex=index_process(incIndex)
# incIndex2=index_process(incIndex2)

x_train[sample_id] = copy.deepcopy(origin_img)
BackgroudFuzz = 0
mutatedResult = originResult
while(mutatedResult==originResult and BackgroudFuzz<180):   
    imgChange(x_train[sample_id], incIndex, 5)
    imgChange(x_train[sample_id], incIndex2, suitableweightfor2)
    mutatedResult = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))      
    BackgroudFuzz += 5

# node_indicator = clf.decision_path(x_train)
# node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                     node_indicator.indptr[sample_id + 1]]

nowdiff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
print(nowdiff)
print(model.predict(np.reshape(x_train[sample_id],(1,28,28,1)))) 
re = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))
baseP = model.predict(np.reshape(x_train[sample_id],(1,28,28,1)))[0][originResult]
nowImg = x_train[sample_id]
plt.imshow(x_train[sample_id].reshape(28,28),cmap='gray')
plt.show()

def get_bin_img(img_index):
    img = np.zeros((28,28))
    for i in img_index:
        img[int(i/28),i%28] = 50
    return img

def imgs_minus(img1, img2):
    for i in range(len(img1)):
        if img1[i]>img2[i]:
            img1[i] = img1[i]-img2[i]
        else:
            img1[i] = 0

# 求出重要性分布图
salience = {}
for p in all_pixels:
    pIndex = []
    addToIndex(pIndex, p)
    img = get_bin_img(pIndex)
    img_flat = np.reshape(img,784)
    changed_img = copy.deepcopy(nowImg)
    imgs_minus(changed_img,img_flat)
    nowP = model.predict(np.reshape(changed_img,(1,28,28,1)))[0][originResult]
    salience[p] = nowP - baseP
    
sorted_salience = sorted(salience.items(), key = lambda kv:(kv[1], kv[0]))
delete_index = []
print(sorted_salience)
for pair in sorted_salience:
    if pair[1]< 0:
        addToIndex(delete_index, pair[0])
     
img = get_bin_img(list(set(delete_index)))
img_flat = np.reshape(img,784)

changed_img = copy.deepcopy(nowImg)

imgs_minus(changed_img,img_flat)
print(model.predict(np.reshape(changed_img,(1,28,28,1))))  
nowdiff=np.linalg.norm(origin_img/255-changed_img/255,ord=2)
print(nowdiff)   
plt.imshow(changed_img.reshape(28,28),cmap='gray')
plt.show()
# np.argsort(salience)

# # 对扰动进行腐蚀操作
imgdiff =  changed_img 
imgs_minus(imgdiff,origin_img)
# tmp = np.reshape(imgdiff,(28,28))


# erode_img(tmp,2)

# imgdiff  = np.reshape(tmp, 28*28)
# plt.imshow(imgdiff.reshape(28,28),cmap='gray')
# plt.show()
change_degree = 0.5
x_train[sample_id] = copy.deepcopy(origin_img)
imgs_add(x_train[sample_id], imgdiff*change_degree)
now_result = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))
while(now_result == originResult and change_degree<2):
    change_degree+=0.05
    x_train[sample_id] = copy.deepcopy(origin_img)
    imgs_add(x_train[sample_id], imgdiff*change_degree)
    now_result = np.argmax(model.predict(np.reshape(x_train[sample_id],(1,28,28,1))))
    nowdiff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
    print(nowdiff)
nowdiff=np.linalg.norm(origin_img/255-x_train[sample_id]/255,ord=2)
print(nowdiff)   
plt.imshow(x_train[sample_id].reshape(28,28),cmap='gray')
plt.show()
print(model.predict(np.reshape(x_train[sample_id],(1,28,28,1)))) 