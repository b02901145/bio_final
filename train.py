import sys
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten

sign_num=20
data_num=50
height=80
width=80

#load training data
train=np.zeros((sign_num*data_num, height*width+sign_num))
for i in range(sign_num):
    for j in range(data_num):
        img=Image.open('result_80_80/'+str(i+1)+'/'+str(i+1)+'_'+str(j+1)+'.jpg')
        train[i*data_num+j, :height*width]=np.array(img).reshape(1, height*width)
    train[i*data_num:i*data_num+data_num, height*width+i]=1

np.random.shuffle(train)
x_train=train[:, :height*width].reshape(sign_num*data_num, 1, height, width)
y_train=train[:, height*width:]
x_train=x_train.astype('float32')
x_train=x_train/255

#convolutional neural network
#define model
model=Sequential()
model.add(Convolution2D(25, 3, 3, input_shape=(1, height, width)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(50, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(sign_num))
model.add(Activation('softmax'))

model.summary()

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit model
model.fit(x_train, y_train, batch_size=100, nb_epoch=10, validation_split=0.2)

#evaluate model
score=model.evaluate(x_train, y_train)
print('Total loss on Training Set:', score[0])
print('Accuracy of Training Set:', score[1])

#save model
model.save('model10.h5')

