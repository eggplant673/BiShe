from imageio import imread
import numpy as np 
image1 = imread('729_4.png')
y = np.reshape(image1,(1,28,28,1))
print(y.shape)
from keras.models import load_model
model = load_model('mnist_model2.h5')
print(model.predict(y))