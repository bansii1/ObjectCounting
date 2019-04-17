import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import keras
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.models import model_from_json
from scipy.misc import imread,imresize
from PIL import Image
PATH=os.getcwd()

data_path=PATH + '/data'
data_dir_list=os.listdir(data_path)

img_rows=120
img_cols=120
num_channel=1
num_epoch=10

num_classes=2

img_data_list=[]
i=0
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+dataset)
	print('Loaded the images of dataset -'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=imread(data_path+'/'+dataset+'/'+img,flatten=True)
		#input_img=Image.open(data_path+'/'+dataset+'/'+img)
		if np.shape(input_img)!=(120,120):
			resized=imresize(input_img,(120,120))
			resized.flatten()
			#resized=imread(resized,flatten=True)
			img_data_list.append(resized)
		else:
			img_data_list.append(input_img)
		
	print(dataset,len(img_data_list))

img_data=np.array(img_data_list)
if num_channel==1:
	if K.image_dim_ordering()=='th':			#theano me channel first format is  used
		img_data= np.expand_dims(img_data, axis=1) 
		print ("expanded:",img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 	#tensorflow me channel last format is used

		print (img_data.shape)
#else:								#for rgb images
#	if K.image_dim_ordering()=='th':
#		img_data=np.rollaxis(img_data,3,1)
print (img_data.shape)
img_data=img_data.astype('float32')	
img_data/=255
print(np.shape(img_data))
# Define the number of classes
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:4334]=1
labels[4335:]=0
	  
names = ['straw','nonstraw']
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

print(np.shape(img_data))
#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
# Defining the model
input_shape=img_data[0].shape
					
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(1,120,120)))
#model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=["accuracy"])
# Training
hist = model.fit(X_train, y_train, batch_size=25, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#serialize model to JSON

model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")

#load JSON and create model
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#score = loaded_model.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

