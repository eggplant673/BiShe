from os import name
import numpy as np
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

def load_data():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train.reshape(len(x_train),28, 28, 1)
  x_test = x_test.reshape(len(x_test),28 ,28, 1)
  y_train = np_utils.to_categorical(y_train, 10)
  y_test = np_utils.to_categorical(y_test, 10)
  return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
  (x_train, y_train), (x_test, y_test) = load_data() 
  model = Sequential()
  #step1：卷积层1 和 池化层1
  model.add(Conv2D(filters=28,kernel_size=(5,5),
                  input_shape=(28, 28, 1), 
                  activation='relu', 
                  padding='same',
                  name = 'cov1'))

  model.add(MaxPooling2D(pool_size=(2, 2))) # 16* 16
  #step2：卷积层2 和 池化层2
  model.add(Conv2D(filters=12, kernel_size=(5, 5), 
                  activation='relu', padding='same',
                  name = 'cov2'))

  model.add(MaxPooling2D(pool_size=(2, 2))) # 8 * 8
  #Step3	建立神经网络(展开层、隐藏层、输出层)
  model.add(Flatten()) # FC1,64个8*8转化为1维向量

  model.add(Dense(784, activation='relu')) # FC2 1024

  model.add(Dense(10, activation='softmax')) # Output 10
  model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

  train_history=model.fit(x_train, y_train,
                          validation_split=0.2,
                          epochs=10, batch_size=256, verbose=1)   
  print(model.predict(x_train)[9])
  model.save('mnist_model2.h5') 