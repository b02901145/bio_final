import cv2
import time
import numpy as np
from PIL import Image

sign_num=20
data_num=50

for i in range(sign_num):
    for j in range(data_num):
        img=Image.open('result_80_80/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg')
	img2=img.transpose(Image.FLIP_LEFT_RIGHT)
	cv2.imwrite('result_mirror/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg',np.array(img))
	cv2.imwrite('result_mirror/'+str(i+1)+'/'+str(i+1)+'_'+str(j+51)+'.jpg',np.array(img2))
