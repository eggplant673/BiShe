from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
# 载入数据集
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
x_img_train = x_img_train.astype('float32') / 255.0
x_img_test = x_img_test.astype('float32') / 255.0
x_img_train_tree = np.array(x_img_train).reshape(len(x_img_train),-1)
# plt.savefig('fix.jpg', dpi=300)
# plt.imshow(x_img_train[1])
# plt.show()


from keras.models import load_model
model = load_model('cifar_10_model.h5')
# aa=model.predict(x_img_train)[10]
# print(aa)

from typing import List, Set
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 决策树生成
# clf=DecisionTreeClassifier()
# clf.fit(x_img_train_tree,np.argmax(model.predict(x_img_train),axis=1))
import joblib
clf = joblib.load('cifar10.pkl')

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
sample_id = 26
# plt.imshow(x_img_train[sample_id])
# plt.show()
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

print('Rules used to predict sample %s: %s' % (sample_id,y_label_train[sample_id]))
print(model.predict(x_img_train)[sample_id])
print(clf.predict(x_img_train_tree)[sample_id])
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

#决策单个图片依赖的像素点
pixelNum = 32*32*3
for node_id in node_index:
    if (x_img_train_tree[sample_id, feature[node_id]] <= threshold[node_id]):
        x_img_train_tree[sample_id, feature[node_id]] = threshold[node_id]+0.8*(1-threshold[node_id])
        if feature[node_id]-33*3 >0 and feature[node_id]+33*3<pixelNum:
           x_img_train_tree[sample_id, feature[node_id]-1*3] ,
           x_img_train_tree[sample_id, feature[node_id]+1*3] ,
           x_img_train_tree[sample_id, feature[node_id]-33*3] ,
           x_img_train_tree[sample_id, feature[node_id]-32*3] ,
           x_img_train_tree[sample_id, feature[node_id]-31*3] ,
           x_img_train_tree[sample_id, feature[node_id]+31*3] ,
           x_img_train_tree[sample_id, feature[node_id]+32*3] ,
           x_img_train_tree[sample_id, feature[node_id]+33*3] = threshold[node_id]+0.8*(1-threshold[node_id])
    else:
        x_img_train_tree[sample_id, feature[node_id]]= threshold[node_id]*0.2
        if feature[node_id]-33*3 >0 and feature[node_id]+33*3<pixelNum:
           x_img_train_tree[sample_id, feature[node_id]-1*3] ,
           x_img_train_tree[sample_id, feature[node_id]+1*3] ,
           x_img_train_tree[sample_id, feature[node_id]-33*3] ,
           x_img_train_tree[sample_id, feature[node_id]-32*3] ,
           x_img_train_tree[sample_id, feature[node_id]-31*3] ,
           x_img_train_tree[sample_id, feature[node_id]+31*3] ,
           x_img_train_tree[sample_id, feature[node_id]+32*3] ,
           x_img_train_tree[sample_id, feature[node_id]+33*3] = threshold[node_id]*0.2
plt.imshow(np.array(x_img_train_tree).reshape(len(x_img_train),32,32,3)[sample_id])
print(model.predict(np.array(x_img_train_tree).reshape(len(x_img_train),32,32,3))[sample_id])
print(clf.predict(x_img_train_tree)[sample_id])
plt.show()


# 找出每个数字决策树偏好判断的像素点
sample_ids = [i for i in range(len(x_img_train_tree)) if clf.predict([x_img_train_tree[i]])==[y_label_train[sample_id]]]
print(len(sample_ids))
pixelNum = 32*32*3
count = [0]*pixelNum 
for id in sample_ids:
    node_index = node_indicator.indices[node_indicator.indptr[id]:
                                    node_indicator.indptr[id + 1]]
    for node_id in node_index:             
        count[feature[node_id]]+=1
index = [i for i in range(pixelNum) if count[i]>=len(sample_ids)*0.40]
img = [0]*pixelNum 
for i in index:
    img[i]=255
print(index)
plt.imshow(np.array(img).reshape(32,32,3))
plt.show()


# # 改变决策分支所依赖的像素点的值
# newIndex = []
# for node_id in index:
#     if(node_id>33 and node_id+33 < pixelNum ):
#         newIndex.append(node_id+1)
#         newIndex.append(node_id-1)
#         newIndex.append(node_id-32)
#         newIndex.append(node_id-33)
#         newIndex.append(node_id-31)
#         newIndex.append(node_id+31)
#         newIndex.append(node_id+32)
#         newIndex.append(node_id+33)     
#         newIndex.append(node_id)  
# for node_id in newIndex:
#         x_img_train_tree[sample_id, node_id] = 0.1
# plt.imshow(x_img_train_tree[sample_id].reshape(32,32,3))
# plt.show()

# print(clf.predict([x_img_train_tree[sample_id]]))
# print(model.predict(np.array(x_img_train_tree).reshape(len(x_img_train),32,32,3))[sample_id])