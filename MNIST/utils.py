from platform import node
from keras.preprocessing import image
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
def init_coverage(path):
    collect = []
    for node in path:
        collect.append(str(node))
    path = ','.join(collect)
    if path not in all_paths:
        all_paths.append(path)
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

def path_convert(path):
    collect = []
    for node in path:
        collect.append(str(node))
    path = ','.join(collect)
    return path

def path_selection(model, path, feature, flag):
    #目标路径选择，这里只是简单的选择最相邻的路径
    next_path = get_next_per(path_convert(path))
    node_index1 = path
    node_index2 = [ int(x) for x in next_path.split(",")]
    # pos = set(node_index2) - set(node_index1)
    neg = set(node_index1) - set(node_index2)
    # neurous_pos = [ feature[node] for node in pos]
    neurous_neg = [ feature[node] for node in neg]
    total_loss = []
    # for index in neurous_pos[0:6]:
    #     total_loss.append(K.mean(model.layers[3].output[...,index]))
    for index in neurous_neg[0:3]:
        total_loss.append(-0.5*K.mean(model.layers[3].output[...,index]))

    for i in range(len(path)-1):
        node = path[i]
        next_node = path[i+1]
        index = feature[next_node]
        if node in path and node not in node_index2 and next_node not in node_index2:
            if flag[i]<0:
                total_loss.append(K.mean(model.layers[3].output[...,index]))
            else:
                total_loss.append(-0.6*K.mean(model.layers[3].output[...,index]))
    # total_loss = []
    # for index in path[:3]:
    #     total_loss.append(K.mean(model.layers[3].output[...,feature[index]]))
    # # for index in neurous_neg:
    # #     grads = get_3rd_layer_output(img)
    # #     grads = np.mean(grads,axis=(1, 2))
    # #     grads = K.gradients(grads[0][index], model.input)[0]
    # #     total_loss.append(-0.2*grads)
    print(np.shape(total_loss))
    return total_loss
    
