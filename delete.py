# import os
# for root,dirs,files in os.walk("C:\\Users\Lenovo\Desktop\shiyan\data\imagenet"):
#    for file in files: 
#       print(os.path.join(root,file))
import numpy as np
vector1 = np.array([1,2,3])
vector2 = np.array([4,5,6])
  
op1=np.sqrt(np.sum(np.square(vector1-vector2)))
op2=np.linalg.norm(vector1-vector2,ord=2)
print(op1)
print(op2)