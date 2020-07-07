from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

from keras.applications.vgg19 import VGG19

import os

os.chdir("C:\\Users\\user\\Documents\\Python\\Deep Learning\\CNN\\VGG19\\PRACTISE")

Model = VGG19(weights ='imagenet')

Model.summary()

Model.save('VGG19.h5')