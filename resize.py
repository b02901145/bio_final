from PIL import Image
import os

sign_num = 20
data_num = 50
width = 80
height = 80

path='./result_'+str(width)+'_'+str(height)
if not os.path.isdir(path):
    os.mkdir(path)

for i in range(sign_num):
    path2='./result_'+str(width)+'_'+str(height)+'/'+str(i+1)
    if not os.path.isdir(path2):
        os.mkdir(path2)
    for j in range(data_num):
        img=Image.open('./result/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg')
        img2=img.resize((width,height))
        img2.save(path+'/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg')
