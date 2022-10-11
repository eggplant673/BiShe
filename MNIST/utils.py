from platform import node
from sysconfig import get_path
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from keras import backend as K

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

all_paths = []
path_to_pos = {}
def init_coverage(path, pos_node):
    collect = []
    for node in path:
        collect.append(str(node))
    path = ','.join(collect)
    if path not in all_paths:
        all_paths.append(path)
        path_to_pos[path] = pos_node
        return True
    return False 

def sort_allpaths():
    all_paths.sort()
    print(len(all_paths))

def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)

covered_paths = []

def path_covered(total_path):
    return covered_paths, total_path, len(covered_paths) / float(total_path)

def update_coverage(path):
    collect = []
    for node in path:
        collect.append(str(node))
    path = ','.join(collect)
    if path not in covered_paths:
        covered_paths.append(path)
        return True
    return False 

def get_next_per(path):
    index = all_paths.index(path)
    if index+1<=len(all_paths):
        return all_paths[index+1]
    else:
        return all_paths[index-1]

def get_next_paths(path):
    index = all_paths.index(path)
    if index+10<=len(all_paths) and index-10>=0:
        return all_paths[index-1:index] + all_paths[index+1:index+2]
    elif index-10<0:
        return all_paths[index+1:index+2]
    else:
        return all_paths[index-1:index]

def path_convert(path):
    collect = []
    for node in path:
        collect.append(str(node))
    path = ','.join(collect)
    return path

def feature_selection(model, x, y, size, importance_map):
    total = []
    m = importance_map

    m = np.mean(m,axis=(0,1))
    for i in range(0,size):
        total.append(m[i]*K.mean(model.layers[3].output[0][...,i]))

    # for i in range(x):
    #     for j in range(y):
    #         for k in range(size):
    #             total.append(m[i,j,k]*K.mean(model.layers[3].output[0][i,j,k]))
    return total

def path_selection(model, path, t_path, feature):
    #目标路径选择，这里只是简单的选择最相邻的路径
    node_index1 = [ int(x) for x in path.split(",")]
    node_index2 = [ int(x) for x in t_path.split(",")]
    path1_pos_node = path_to_pos[path]
    path2_pos_node = path_to_pos[t_path]

    pos = set(node_index2) - set(node_index1)
    neg = set(node_index1) - set(node_index2)
    total_loss = []
  
    for index in pos:
        if index in path2_pos_node:
            total_loss.append(K.mean(model.layers[3].output[...,feature[index]]))
        else:
            total_loss.append(-K.mean(model.layers[3].output[...,feature[index]]))
    for index in neg:
        if index in path1_pos_node:
            total_loss.append(-K.mean(model.layers[3].output[...,feature[index]]))
        else:
            total_loss.append(K.mean(model.layers[3].output[...,feature[index]]))
    return total_loss

feature_to_importance = {}