# Tendorflow imports
import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# Helper libraries
import math
import numpy as np
import matplotlib
matplotlib.use('PS') #prevent import error due to venv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Imports for dataset separation
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# allow for dataset iteration. 
#tf.enable_eager_execution() #comment this out if causing errors

###       GET THE DATASET AND SOME INFO ABOUT IT       ###
# get the data into slices
data_images = []
data_labels = []
rel_img_path = 'map-proj/' # add path of folder to image name for later loading

# open up the labeled data file
with open('labels-map-proj.txt') as labels:
  for line in labels:
    file_name, label = line.split(' ')
    data_images.append(rel_img_path + file_name)
    data_labels.append(int(label))

# divide data into testing and training (total len 3820)
train_images, test_images, train_labels, test_labels = train_test_split(
    data_images, data_labels, test_size=0.15, random_state=666)
test_len = len(test_images)   # 573
train_len = len(train_images) # 3247

# label translations
class_labels = ['other','crater','dark_dune','streak',
                'bright_dune','impact','edge']


###                PREPROCESS THE DATA                 ###
#convert image paths into numpy matrices
def parse_image(filename):
  img_obj = Image.open(filename)
  img = np.asarray(img_obj).astype(np.float32)
  #normalize image to 0-1 range
  img /= 255.0
  return img

train_images = np.array(list(map(parse_image, train_images)))
test_images = np.array(list(map(parse_image, test_images)))

# convert labels to one-hot encoding
def to_one_hot(label):
  encoding = [0 for _ in range(len(class_labels))]
  encoding[label] = 1
  return np.array(encoding).astype(np.float32)

train_labels = np.array(list(map(to_one_hot, train_labels)))
test_labels = np.array(list(map(to_one_hot, test_labels)))

# model.fit requires train data to be in the shape of [batch, imDim1, imDim2, numChannels]
# slap extra dimension on the end of train images so tf will be happy
train_images = np.reshape(train_images, (-1, 227, 227, 1)) #add 4th dim
train_labels = np.reshape(train_labels, (-1, 7))
test_images = np.reshape(test_images, (-1, 227, 227, 1))
test_labels = np.reshape(test_labels, (-1, 7))

# make a generator to train the model with
generator = ImageDataGenerator(rotation_range=0, zoom_range=0,
    width_shift_range=0, height_shift_range=0, shear_range=0,
    horizontal_flip=False, fill_mode="nearest")


###             BUILD SHAPE OF THE MODEL              ###
# increase kernel size and stride??
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
      input_shape=(227,227,1)),
  tf.keras.layers.MaxPooling2D((2,2), strides=2),
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D((2,2), strides=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(7, activation=tf.nn.softmax), # final layer with node for each classification
])

# specify loss and SGD functions
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


###                 TRAIN THE MODEL                   ###
#specify training metadata
BATCH_SIZE = 32
print("about to train")
# train the model on the training data
num_epochs = 5 
model.fit_generator(generator.flow(train_images, train_labels, batch_size=BATCH_SIZE), epochs=num_epochs)

###             EVALUATE MODEL ACCURACY               ###
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Final loss was {}.\nAccuracy of model was {}".format(test_loss,test_accuracy))


