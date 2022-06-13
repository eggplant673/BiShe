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

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

rows=[]
title=["beast","head_of_junco","head_not_of_junco","torso","result"]
for root,dirs,files in os.walk("D:\\shiyan\data\\test"):
    for img_path in files: 
        img = image.load_img(os.path.join(root,img_path), target_size=(224, 224))   # 大小为224*224的Python图像库图像
        x = image.img_to_array(img)  # 形状为（224， 224， 3）的float32格式Numpy数组、
        x = np.expand_dims(x, axis=0)  # 添加一个维度，将数组转化为（1， 224， 224， 3）的形状批量
        x = preprocess_input(x)   #按批量进行预处理（按通道颜色进行标准化）

        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])
 
        #为了展示图像中哪些部分最像非洲象，我们使用Grad-CAM算法：
        african_elephant_output = model.output[:, np.argmax(preds[0])]   # 预测向量中的元素
        last_conv_layer = model.get_layer('block5_conv3')  # block5_conv3层的输出特征图，它是VGG16的最后一个卷积层
        grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]   # 类别相对于block5_conv3输出特征图的梯度
        pooled_grads = K.mean(grads, axis=(0, 1, 2))   # 形状是（512， ）的向量，每个元素是特定特征图通道的梯度平均大小
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])  # 这个函数允许我们获取刚刚定义量的值：对于给定样本图像，pooled_grads和block5_conv3层的输出特征图
        pooled_grads_value, conv_layer_output_value = iterate([x])  # 给我们两个样本图像，这两个量都是Numpy数组

        origin_conv = conv_layer_output_value
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

        # filter_Mean = K.mean(last_conv_layer.output,axis=(0,1,2))
        # filter_Index = K.argmax(filter_Mean) 
        # inputgrads = K.gradients(K.mean(last_conv_layer.output[0][:,:,487]) ,model.input)[0]
        # fuzzling = K.function([model.input],[inputgrads])
        # fuzzed = fuzzling([x])
     
        # # fuzzed = np.reshape(fuzzed,(224,224,3))
        # x = x + 6000*np.array(fuzzed)
        # print(6000*np.max(fuzzed))
        # preds_1 = model.predict(x[0])

        # print('Predicted:', decode_predictions(preds_1, top=3)[0])


        pool_mean = np.mean(conv_layer_output_value,axis=(0, 1))
        filter_number = np.argsort(pool_mean)[-1]

        row = []
        for i in [167,389,487,501]:
            row.append(pool_mean[i])
        # for i in [167,389,487,501]:
        #     if pool_mean[i]>0.00002:
        #         row.append(1)
        #     else:
        #         row.append(0)
        if decode_predictions(preds, top=3)[0][0][1] == "junco":
            row.append("junco")
        else:
            row.append("not junco")
        rows.append(row)

        # # heatmap = np.mean(origin_conv, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图
        # heatmap = origin_conv[:,:,167] # 单独特征热力图
        # conv_layer_output_value = np.maximum(conv_layer_output_value,0)

        # print(np.argsort(pool_mean)[-1:-7:-1])
        # indexes = np.argsort(pool_mean)[-1:-7:-1]
        # # print(pool_mean[indexes[0]])
        # # print(pool_mean[indexes[1]])
        # # print(pool_mean[indexes[2]])
        # # print(pool_mean[indexes[3]])
        # # print(pool_mean[indexes[4]])
        # # print(pool_mean[indexes[5]])
        # heatmap = np.maximum(heatmap, 0)
        # heatmap /= np.max(heatmap)+1e-5

        # img1 = cv2.imread(os.path.join(root,img_path))  # 用cv2加载原始图像
        # img1 = cv2.resize(img1,dsize=(224,224))

        # heatmap = cv2.resize(heatmap, (img1.shape[1], img1.shape[0]))  # 将热力图的大小调整为与原始图像相同
        # heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # 将热力图应用于原始图像
    
        # superimposed_img1 = cv2.addWeighted(img1,1,heatmap,0.4,0)
        # cv2.imwrite('D:\\shiyan\data\imagenet_cam\cub_0\\'+img_path+'-cam.png', superimposed_img1)
        # print('----------------------------')

import csv
with open("datasets/junco_regression_test.csv",'w',newline='') as t:#numline是来控制空的行数的
    writer=csv.writer(t)#这一步是创建一个csv的写入器
    writer.writerow(title)#写入标签
    writer.writerows(rows)#写入样本数据
    