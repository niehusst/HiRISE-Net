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

# Imports for dataset separation
from sklearn.model_selection import train_test_split

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# allow for dataset iteration. 
tf.enable_eager_execution() #comment this out if causing errors

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
test_len = tf.cast(len(data_labels) * 0.15, tf.int64)  # 573
train_len = tf.cast(len(data_labels) * 0.85, tf.int64) # 3247
train_images, test_images, train_labels, test_labels = train_test_split(
    data_images, data_labels, test_size=0.15, random_state=666)


# label translations
class_labels = ['other','crater','dark_dune','streak',
                'bright_dune','impact','edge']


###                PREPROCESS THE DATA                 ###
# convert train and test data to tf datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

#convert image paths into real images
def parse_image(filename, label):
  im_string = tf.read_file(filename)
  im_decoded = tf.image.decode_jpeg(im_string, channels=1)
  img = tf.cast(im_decoded, tf.float32)
  label = tf.cast(label, tf.int32)
  return img, label

# convert to 0-1 range
def normalize(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

train_dataset.map(parse_image)
train_dataset.map(normalize)
test_dataset.map(parse_image)
test_dataset.map(normalize)


###             BUILD SHAPE OF THE MODEL              ###
# increase kernel size and stride??
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
      input_shape=(227,227,1)),
  tf.keras.layers.MaxPooling2D((2,2), strides=2),
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D((2,2), strides=2),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  tf.keras.layers.Dense(7, activation=tf.nn.softmax) # final layer with node for each classification
])

# specify loss functions
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

###                 TRAIN THE MODEL                   ###
#specify training metadata
BATCH_SIZE = 32
# shuffled so nothing can be learned from the order of the data
train_dataset = train_dataset.repeat().shuffle(train_len).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# train the model on the training data
num_epochs = 1 #TODO: increase later
model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=math.ceil(train_len/BATCH_SIZE))


###             EVALUATE MODEL ACCURACY               ###
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_len/BATCH_SIZE))
print("Final loss was {}.\nAccuracy of model was {}".format(test_loss,test_accuracy))


