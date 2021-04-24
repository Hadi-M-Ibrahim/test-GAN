import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\hadis stuff\AI project data\wikiart',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode="rgb",
    batch_size = batch_size,
    image_size=(img_height, img_width),
    shuffle = True,
    seed = 123,
    validation_split= 0.2,
    subset = "training"
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\hadis stuff\AI project data\wikiart',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode="rgb",
    batch_size = batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    seed = 123,
    validation_split= 0.2,
    subset = "validation"
)

class_names = train_ds.class_names
print(class_names)