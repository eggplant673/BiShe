import os
for root,dirs,files in os.walk("C:\\Users\Lenovo\Desktop\shiyan\data\imagenet"):
   for file in files: 
      print(os.path.join(root,file))