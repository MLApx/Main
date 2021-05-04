#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt


# In[25]:
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model = models.Sequential()
model.add(layers.experimental.preprocessing.Rescaling(
    1./255, offset=0.0, name=None,input_shape=(512, 512, 3)
))
model.add(layers.Conv2D(16, (5,5),kernel_initializer=initializer,kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001), activation='relu', input_shape=(512, 512, 3)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(32, (5, 5),kernel_initializer=initializer, kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001),activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3),kernel_initializer=initializer,kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5),kernel_initializer=initializer,kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5),kernel_initializer=initializer,kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.2))
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(64, activation='softmax'))
model.add(layers.Dense(16,kernel_initializer=initializer,kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0001), activation='relu'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(1,kernel_initializer=initializer,kernel_regularizer=l2(0.0001),bias_regularizer=l1(0.0001),activation='sigmoid'))
model.summary()


# In[21]:


train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    '/deac/classes/egr315/tongh18/train/20x/MN/',
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=1337,
    interpolation="bilinear",
    follow_links=False,
    validation_split=0.2,
    subset="training"
)
val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    '/deac/classes/egr315/tongh18/train/20x/MN/',
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=1337,
    interpolation="bilinear",
    follow_links=False,
    validation_split=0.2,
    subset="validation"
)


# In[22]:

class_names=train_ds.class_names
print(class_names)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


# In[26]:

class_weights={0:7.,1:1.}
model.compile(loss='binary_crossentropy', metrics=['accuracy'])


# In[28]:


history = model.fit(
    train_ds, epochs=15, validation_data=val_ds,shuffle=True,verbose=2,workers=2,use_multiprocessing=True,class_weight=class_weights
)
model.save("/deac/classes/egr315/tongh18/20xbinarymodelMN02.h5")

# In[ ]:


print(history.history.keys())
# summarize history for accuracy
'''
plt.close()
plt.ioff()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig('/deac/classes/egr315/tongh18/test1/accuracy40.png')
plt.savefig('/deac/classes/egr315/tongh18/accuracy20xbinaryMN.png')
plt.close()
# summarize history for loss
plt.ioff()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/deac/classes/egr315/tongh18/test1/loss20xbinaryMN.png')
plt.close()
'''
import os
from PIL import Image

LAMN_ds=tf.keras.preprocessing.image_dataset_from_directory(
    '/deac/classes/egr315/tongh18/test_set/20x/MN/LAMN/',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=1337,
    interpolation="bilinear",
    follow_links=False,
)
z=model.predict(LAMN_ds)
znp=np.where(z>0.5,1,0)
zcount=np.unique(znp,return_counts=True)
print(z)
print(zcount)
HAMN_ds=tf.keras.preprocessing.image_dataset_from_directory(
    '/deac/classes/egr315/tongh18/test_set/20x/MN/HAMN',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=1337,
    interpolation="bilinear",
    follow_links=False,
)
zz=model.predict(HAMN_ds)
zznp=np.where(zz>0.5,1,0)
zzcount=np.unique(zznp,return_counts=True)
print(zz)
print(zzcount)

