import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn import svm
from keras.models import load_model
from scipy.ndimage import label

model=load_model('model_m.h5')
sign_type=['1-one','2-two','3-three','4-four','5-five',
           '6-six','7-seven','8-eight','9-nine','10-hand',
           '11-square','12-child','13-money','14-dragon','15-wc',
           '16-wait','17-hundred','18-fourty','19-zero','20-borrow'];
font=ImageFont.truetype("arial.ttf",20)

def myArray2Image(A, dataType):
    A=A.astype('uint8')
    A=Image.fromarray(A, dataType)
    return A


def main(A1,img5,Y1,fr,fc,run):
    Y1=Y1[np.amin(fr):np.amax(fr),np.amin(fc):np.amax(fc)]
    [height, width]=Y1.shape

    max_num=max(height,width)
    max_num=(np.ceil(max_num/2.0)*2).astype('uint8')
    Y2=np.zeros((max_num,max_num))

    temp1=(np.ceil(height/2)).astype('uint8')
    temp2=(np.ceil(width/2)).astype('uint8')
    
    Y2[(max_num/2-temp1):(max_num/2-temp1+height),
       (max_num/2-temp2):(max_num/2-temp2+width)]=Y1

    Y2=myArray2Image(Y2,'L')
    Y2=Y2.resize((80,80))
    
    Y2=np.array(Y2)
    data=Y2
    result=model.predict(data.reshape(1, 1, 80, 80))
    result=np.argmax(result, axis=1)+1

    A1=myArray2Image(A1,'L')
    A1=np.array(A1.resize((640,480)))
    tmp=(A1>0)

    tmp2=np.ones(shape=(480,640))
    img5=img5.astype('float64')
    if run==0:
        img5[:,:,0]*=(tmp2*0.5+tmp*0.5)
        img5[:,:,1]*=(tmp2*0.5+tmp*0.5)
        img5[:,:,2]*=(tmp2*0.5+tmp*0.5)
    else:
	img5[:,:,0]*=tmp2
        img5[:,:,1]*=tmp2
        img5[:,:,2]*=tmp2
    fr_min=np.amin(fr*480/135)
    fr_max=np.amax(fr*480/135)
    fc_min=np.amin(fc*480/135)
    fc_max=np.amax(fc*480/135)

    img5[fr_min:fr_min+2,fc_min:fc_max+1,:]=[0,0,255]
    img5[fr_max-1:fr_max+1,fc_min:fc_max+1,:]=[0,0,255]
    img5[fr_min:fr_max+1,fc_min:fc_min+2,:]=[0,0,255]
    img5[fr_min:fr_max+1,fc_max-1:fc_max+1,:]=[0,0,255]

    img5=myArray2Image(img5,'RGB')
    draw=ImageDraw.Draw(img5)
    draw.text((fc_min,fr_min-20),sign_type[result-1],(0,0,255),font=font)
    draw=ImageDraw.Draw(img5)
    img5=np.array(img5)
    print sign_type[result-1]
    return img5


#train
feature_1=np.zeros((24,3))
for i in range(24):
    if i<9:
        img=Image.open('tr/tr0'+str(i+1)+'.jpg')
    else:
        img=Image.open('tr/tr'+str(i+1)+'.jpg')
    img=np.array(img.convert('YCbCr'))
    feature_1[i,0]=np.mean(img[:,:,0])
    feature_1[i,1]=np.mean(img[:,:,1])
    feature_1[i,2]=np.mean(img[:,:,2])

label_1=np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,1,1,0,1])

#feature scaling
mean=np.mean(feature_1,axis=0)
std=np.std(feature_1,axis=0)
feature_1-=mean
feature_1/=std

#svm
clf=svm.SVC()
clf.fit(feature_1, label_1)

#capture frame
cap=cv2.VideoCapture(0)


#test
while(True):
    #capture frame-by-frame
    t_start=time.time()
    ret, frame=cap.read()
    img2=np.array(frame)
    img5=img2
    img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    img2=myArray2Image(img2,'RGB')
    img2=img2.resize((180,135))
    img2=np.array(img2.convert('YCbCr'))
    
    #downsampling
    Y=img2[:,:,0]

    M=np.size(img2,0)
    N=np.size(img2,1)
    img2=img2.reshape(M*N,3)

    #feature scaling
    img2=img2.astype('float64')
    img2-=mean
    img2/=std

    #test
    img2=clf.predict(img2).reshape(M,N)

    #group the regions
    s=np.array([[1,1,1],
                [1,1,1],
                [1,1,1]])
    A1, num=label(img2,structure=s)

    #find the size of each region
    size1=np.zeros((num,1))
    for i in range(num):
        size1[i]=np.where(A1==i+1)[0].shape

    #find the largest three regions
    if size1.shape[0]==0:
	continue
    I1=np.argmax(size1)
    size_max1=np.amax(size1)
    size1[I1]=0
    I2=np.argmax(size1)
    size_max2=np.amax(size1)
    size1[I2]=0
    I3=np.argmax(size1)
    size_max3=np.amax(size1)

    [fr1, fc1]=np.where(A1==I1+1)
    [fr2, fc2]=np.where(A1==I2+1)
    [fr3, fc3]=np.where(A1==I3+1)
    
    BB=np.array([np.mean(fr1),np.mean(fr2),np.mean(fr3)])
    BB2=np.array([np.mean(fr1),np.mean(fr2),np.mean(fr3)])
    fr_array=np.array([fr1,fr2,fr3])
    fc_array=np.array([fc1,fc2,fc3])
    I_array=np.array([I1,I2,I3])

    big=np.argmax(BB)
    BB[big]=0
    middle=np.argmax(BB)
    BB[middle]=0
    small=np.argmax(BB)

    #three regions are larger than 400: choose the lower two
    if size_max3>=400:
	if np.mean(fr1)==BB2[small]:
	    AA1=Y*(A1==I2+1)
	    AA2=Y*(A1==I3+1)
	    img5=main(A1,img5,AA1,fr2,fc2,0)
	    img5=main(A1,img5,AA2,fr3,fc3,1)
	elif np.mean(fr2)==BB2[small]:
	    AA1=Y*(A1==I1+1)
	    AA2=Y*(A1==I3+1)
	    img5=main(A1,img5,AA1,fr1,fc1,0)
	    img5=main(A1,img5,AA2,fr3,fc3,1)
        else:
	    AA1=Y*(A1==I1+1)
	    AA2=Y*(A1==I2+1)
	    img5=main(A1,img5,AA1,fr1,fc1,0)
	    img5=main(A1,img5,AA2,fr2,fc2,1)

    #two regions are larger than 400: choose the lower one
    elif size_max3<400 and size_max2>=400:
	if BB2[middle]<400:
	    AA1=Y*(A1==I_array[big]+1)
	    img5=main(A1,img5,AA1,fr_array[big],fc_array[big],0)
	else:
	    AA1=Y*(A1==I_array[middle]+1)
	    img5=main(A1,img5,AA1,fr_array[middle],fc_array[middle],0)

    #one or no region is larger than 400
    else:
	A1=myArray2Image(A1,'L')
        A1=np.array(A1.resize((640,480)))
	tmp=(A1>0)
	tmp2=np.ones(shape=(480,640))
	img5=img5.astype('float64')
	img5[:,:,0]*=(0.5*tmp+0.5*tmp2)
	img5[:,:,1]*=(0.5*tmp+0.5*tmp2)
	img5[:,:,2]*=(0.5*tmp+0.5*tmp2)
	img5=myArray2Image(img5,'RGB')
	img5=np.array(img5)

    cv2.imshow('frame',img5)

    t_end=time.time()
    print t_end-t_start

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

