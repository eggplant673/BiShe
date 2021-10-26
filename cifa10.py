from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
# 载入数据集
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape) 
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape) 
# 归一化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
# One-Hot Encoding
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# 搭架子
model = Sequential()
#step1：卷积层1 和 池化层1
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) # 16* 16
#step2：卷积层2 和 池化层2
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) # 8 * 8
#Step3	建立神经网络(展开层、隐藏层、输出层)
model.add(Flatten()) # FC1,64个8*8转化为1维向量
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu')) # FC2 1024
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax')) # Output 10
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=10, batch_size=128, verbose=1)   
print(model.predict(x_img_train)[9])
# model.save('cifar_10_model.h5') 

