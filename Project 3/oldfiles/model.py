#################################
# Part 1: Loading required data #
#################################

# Part 1.1: Load libraries

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import csv
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D

# Part 1.2: Loading csv lines

lines1 = [] # arrays for loading lines
lines2 = []
lines3 = []
lines4 = []
lines5 = []
alllines = []

with open('data/u/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0]='data/u/'+line[0][:]  # center image location
        line[1]='data/u/'+line[1][1:] # left image location
        line[2]='data/u/'+line[2][1:] # right image location
        lines1.append(line)
lines1 = np.delete(lines1, 0, 0) # delete 1st row with titles
alllines.append(lines1)

with open('data/f1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0]='data/f1/'+line[0][20:]
        line[1]='data/f1/'+line[1][20:]
        line[2]='data/f1/'+line[2][20:]
        lines2.append(line)
lines2 = np.delete(lines2, 0, 0)
alllines.append(lines2)

with open('data/f2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0]='data/f2/'+line[0][20:]
        line[1]='data/f2/'+line[1][20:]
        line[2]='data/f2/'+line[2][20:]
        lines3.append(line)
lines3 = np.delete(lines3, 0, 0)

with open('data/f3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0]='data/f3/'+line[0][20:]
        line[1]='data/f3/'+line[1][20:]
        line[2]='data/f3/'+line[2][20:]
        lines4.append(line)
lines4 = np.delete(lines4, 0, 0)

with open('data/f4/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader:
        line[0]='data/f4/'+line[0][20:]
        line[1]='data/f4/'+line[1][20:]
        line[2]='data/f4/'+line[2][20:]
        lines5.append(line)
lines5 = np.delete(lines5, 0, 0)

alllines=np.concatenate((lines1,lines2,lines3,lines4,lines5)) # concatenate all data

print("Data Loaded")


############################
# Part 2: Transform images #
############################

def prepare_image(img):
    img2 = img[65:125,:,:]	# crop image's top 65 and bottom 35 pixels
    img3 = cv2.resize(img2, (160, 30))  # resize from 320x60 to 160x30
    img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2YUV) # convert to YUV color space
    return img4 

def flip_image(img):
    return cv2.flip(img, 1) # flip image horizontally


images = []
measurements = []

for line in lines5:
    image = cv2.imread(line[0]) 	# read center image 
    inormal = prepare_image(image)	# crop and convert image
    images.append(inormal)
    m = float(line[3])			# read steering measurement for the image
    measurements.append(m)
    iflipped=flip_image(inormal)	# flip image for augmentation
    images.append(iflipped)
    measurements.append(-m)		# negative steering for flipped image
    
    image = cv2.imread(line[1])		# read left image 
    inormal = prepare_image(image)
    images.append(inormal)
    measurements.append(m+0.2)		# add 0.2 correction for return

    image = cv2.imread(line[2])		# read right image 
    inormal = prepare_image(image)
    images.append(inormal)
    measurements.append(m-0.2)		# subtract 0.2 correction for return

x_train = np.array(images)
y_train = np.array(measurements)
print("Images and measurements Loaded")


##############################
# Part 3: The Training model #
##############################

# This is a modified Nvidia's architecture implementation

#defining model:
model = Sequential()

# adding normalization:
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(30,160,3)))

# adding layers: 5 convolutional and 4 fully-connected

model.add(Convolution2D(filters=24,kernel_size=(3,3),strides=(2, 2), activation='relu'))
#output size: (14,79,24)

model.add(Convolution2D(filters=36,kernel_size=(3,3),strides=(2, 2), activation='relu'))
#output size: (6,39,36)

model.add(Convolution2D(filters=48,kernel_size=(2,2),strides=(2, 2), activation='relu'))
#output size: (3,19,48)

model.add(Convolution2D(filters=64,kernel_size=(2,2),strides=(1, 1), activation='relu'))
#output size: (2,18,64)

model.add(Convolution2D(filters=64,kernel_size=(2,2),strides=(1, 1), activation='relu'))
#output size: (1,17,64)

model.add(Flatten())
#output size: 17*64

model.add(Dense(100))
#output size: 100

model.add(Dense(50))
#output size: 50

model.add(Dense(10))
#output size: 10

model.add(Dense(1))
#output size: 1

adamandeve = optimizers.Adam(lr=0.0004)
# using adam optimizer with low learning rate

model.compile(optimizer=adamandeve, loss='mse')
# using mean square error since the output desired is like regression 

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
# training the model

model.save('model.h5')

print("Model Trained")
