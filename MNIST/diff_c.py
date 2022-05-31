import numpy as np
import os
from utils import *

img_dir = './seeds_50'
tmp_img = preprocess_image(os.path.join(img_dir,'4413_5.png'))
l2_o = np.linalg.norm(tmp_img)

gen_dir2 = './generated_inputs/0602'
tmp_img2 = preprocess_image(os.path.join(gen_dir2,'4413_5_94519088313.png'))
gen_dir3 = './generated_inputs/new'
tmp_img3 = preprocess_image(os.path.join(gen_dir3,'4413_5_6_0.32580498.png'))
l2_2 = np.linalg.norm(tmp_img2)
l2_3 = np.linalg.norm(tmp_img3)
print(l2_2/l2_o)
print(l2_3/l2_o)