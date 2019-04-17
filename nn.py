import numpy as np
import time
import cv2
import os,os.path
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import csv 
import random
import matplotlib.pyplot as plt
import tensorflow as tf
#following code shows writing image names into a file and shuffling it
f=open("oput.csv",'r+') #create an empty file first of same name
w=csv.writer(f)
for path, dirs, files in os.walk("p"):
    for filename in files:
        w.writerow([filename])
for path,dirs,files in os.walk("n"):
    for filename in files:
        w.writerow([filename])
mixlist=[]

'''with open('weights.csv','r') as f:
	reader = csv.reader(f,delimiter=',')
	mixlist=mixlist+list(reader)
copy=mixlist
print(mixlist)
del mixlist[0]
random.shuffle(mixlist)
random.shuffle(mixlist)
random.shuffle(mixlist)
random.shuffle(mixlist)
random.shuffle(mixlist)
random.shuffle(mixlist)
print(mixlist)
df = pd.DataFrame(mixlist)
df.to_csv("mixdata.csv")'''

seed=128
rng=np.random.RandomState(seed)
rawtrain=pd.read_csv("mixdata.csv")
train=rawtrain[:-228]
rawtest=pd.read_csv("mixdata.csv")
rawtest=rawtest[914:]
keep_col=['filename']
#to retain onlyl filename column in test dataset
print(rawtest[keep_col])
test=rawtest[keep_col]

#randomly selects a filename from training data
img_name=rng.choice(train.filename)
filepath=os.path.join("mix",img_name)

img=imread(filepath,flatten=True)

#plt.imshow(img,cmap='gray')
#plt.show()

temp=[]
for image_name in train.filename:
    image_path=os.path.join("mix",img_name)
    img=imread(image_path,flatten=True)
    img=img.astype('float32')
    temp.append(img)

print(np.shape(temp))
train_x=np.stack(temp)
print(np.shape(train_x))

temp = []
for img_name in test.filename:
    image_path = os.path.join("mix",img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
test_x = np.stack(temp)

#splitting in 7:3 ratio of train set for train and validation set
split_size=int(train_x.shape[0]*0.7)

train_x,val_x = train_x[:split_size],train_x[split_size:]
train_y,val_y = train.label.values[:split_size],train.label.values[split_size:]

def dense_to_one_hot(labels_dense,num_classes=2):
#convert class tables from scalars to one-hot vectors
    num_labels=labels_dense.shape[0]
    index_offset=np.arange(num_labels)*num_classes          
    labels_one_hot=np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()]=1
   
    return labels_one_hot


def preproc (unclean_batch_x):
#convert values into 0 and 1
    temp_batch=unclean_batch_x/unclean_batch_x.max()
    return temp_batch    

def batch_creator(batch_size,dataset_length,dataset_name):
#create batch with random samples and return appropriate format
    batch_mask=rng.choice(dataset_length,batch_size)
    batch_x=eval(dataset_name+'_x')[[batch_mask]].reshape(-1,input_num_units)
    batch_x=preproc(batch_x)

    if dataset_name=='train':
        batch_y = eval(dataset_name).ix[batch_mask,'label'].values
        batch_y = dense_to_one_hot(batch_y)

    return batch_x,batch_y

#number of neurons in each layer
input_num_units=120*120 #size of image
hidden_num_units=600  #can vary
output_num_units=2  #two classes,1 for positve and 0 for negative

#defining placeholders

x=tf.placeholder(tf.float32,[None,input_num_units])
y=tf.placeholder(tf.float32,[None,output_num_units])

#set remaining variables

epochs=10
batch_size=128
learning_rate=0.001

#defining weights and biases of nn

weights={
'hidden':tf.Variable(tf.random_normal([input_num_units , hidden_num_units], seed=seed)), 
'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units],seed=seed))
}

biases={
'hidden': tf.Variable(tf.random_normal([hidden_num_units],seed=seed)),
'output': tf.Variable(tf.random_normal([output_num_units],seed=seed))
}

#defining neural netwrok

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

#defining cost

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels= y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#init_all_variables() is deprecated
init = tf.global_variables_initializer()



with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")
    
    
    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print ("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))
    
    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})



#test_index is index of test image from test set that you wanna predict
for test_index in range(len(test_x)):
#img = img.astype('float32')
    print(np.shape(test_x[test_index]))
    print( "Prediction is: ", pred[test_index])

    plt.imshow(test_x[test_index],cmap='gray')

    plt.show()
    time.sleep(2)



