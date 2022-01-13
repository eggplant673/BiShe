from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os 
tf.compat.v1.disable_eager_execution()

K.clear_session()
model = VGG16(weights='imagenet')

for root,dirs,files in os.walk("C:\\Users\Lenovo\Desktop\shiyan\data\imagenet\pre"):
    for img_path in files: 
        img = image.load_img(os.path.join(root,img_path), target_size=(224, 224))   # 大小为224*224的Python图像库图像
        x = image.img_to_array(img)  # 形状为（224， 224， 3）的float32格式Numpy数组、
        x = np.expand_dims(x, axis=0)  # 添加一个维度，将数组转化为（1， 224， 224， 3）的形状批量
        x = preprocess_input(x)   #按批量进行预处理（按通道颜色进行标准化）

        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])

        print('网络认为预测向量中最大激活类别的元素，索引编号:',np.argmax(preds[0]))

        #为了展示图像中哪些部分最像非洲象，我们使用Grad-CAM算法：
        african_elephant_output = model.output[:, np.argmax(preds[0])]   # 预测向量中的非洲象元素
        last_conv_layer = model.get_layer('block5_conv3')  # block5_conv3层的输出特征图，它是VGG16的最后一个卷积层
        grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]   # 非洲象类别相对于block5_conv3输出特征图的梯度
        pooled_grads = K.mean(grads, axis=(0, 1, 2))   # 形状是（512， ）的向量，每个元素是特定特征图通道的梯度平均大小
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])  # 这个函数允许我们获取刚刚定义量的值：对于给定样本图像，pooled_grads和block5_conv3层的输出特征图
        pooled_grads_value, conv_layer_output_value = iterate([x])  # 给我们两个大象样本图像，这两个量都是Numpy数组

        # for i in range(512):
        #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

        size = 224
        margin = 5

        pool_mean = np.mean(conv_layer_output_value,axis=(0, 1))
        filter_number = np.argsort(pool_mean)[-1]

        # # This a empty (black) image where we will store our results.
        # results = np.zeros((16 * size + 16 * margin, 32 * size + 32 * margin))# numpy数组来存放结果，结果与结果之间有黑色的边际

        # for i in range(16):  # iterate over the rows of our results grid，行循环
        #     for j in range(32):  # iterate over the columns of our results grid，列循环
        #         # Generate the pattern for filter `i + (j * 8)` in `layer_name`
        #         heatmap = np.maximum(conv_layer_output_value[:, :, i+16*j], 0)
        #         heatmap /= np.max(heatmap)+ 1e-5
        #         filter_img =cv2.resize(heatmap, (224, 224))
        #         heatmap = np.uint8(255 * heatmap)
        #         # Put the result in the square `(i, j)` of the results grid
        #         horizontal_start = i * size + i * margin# 水平方向开始的调整
        #         horizontal_end = horizontal_start + size# 水平方向结束的调整
        #         vertical_start = j * size + j * margin# 垂直开始方向的调整
        #         vertical_end = vertical_start + size# 垂直结束方向的调整
        #         results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img# 最终所有过滤器的可视化结果

        # # Display the results grid

        # plt.imshow(results)
        # plt.show()


        # heatmap = np.mean(conv_layer_output_value, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图
        heatmap = conv_layer_output_value[:,:,487]
        print(np.argsort(pool_mean)[-1:-5:-1])
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)+1e-5

        img = cv2.imread(os.path.join(root,img_path))  # 用cv2加载原始图像

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img    # 这里的0.4是热力图强度因子
        #plt.imshow(superimposed_img)
        #plt.show()
        cv2.imwrite('C:\\Users\Lenovo\Desktop\shiyan\data\imagenet_cam\pre\\'+img_path+'-cam.png', superimposed_img)   # 将图像保存到硬盘