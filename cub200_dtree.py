from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import joblib
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, plot_tree

tf.compat.v1.disable_eager_execution()

K.clear_session()
model = VGG16(weights='imagenet')


def convert_to_float(data):
    rlist = []
    for line in data:
        line = list(map(float,line))
        rlist.append(line)
    return rlist


def load_data(data_file, parse_attributes=False):
    data = [[]]
    label = []
    with open(data_file) as f:
        for line in f:
            line = line.strip("\r\n")
            line = line.split(',')
            data.append(line[:-1])
            label.append(line[-1])
    data.remove([])

    if parse_attributes:
        attributes = data[0]
        data.pop(0)  # remove attributes from it
        label.pop(0)
        data = convert_to_float(data)
        return data, label, attributes
    else:
        data = convert_to_float(data)
        return data, label


x_train, y_train, attributes = load_data("datasets/junco_regression.csv", True)
x_train = np.array(x_train)
clf=DecisionTreeClassifier()
print(np.shape(x_train))
clf.fit(x_train,y_train)

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold


node_indicator = clf.decision_path(x_train)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = clf.apply(x_train)

# HERE IS WHAT YOU WANT
sample_id = 6

node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]


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

print(np.shape(leave_id))
x_test, y_test = load_data("datasets/junco_regression.csv")
result_test = clf.predict(x_test)
count = 0
for i in range(np.size(y_test)):
    if result_test[i] == y_test[i]:
        count = count+1
print("%d total test cases, %d true" % (np.size(y_test), count))

from sklearn import tree
tree.plot_tree(clf,
               feature_names = ["chest","head_junco","head_not_junco","torso"], 
               class_names=["junco","bunting"],
               rounded=True, 
               filled = True)

plt.show()

