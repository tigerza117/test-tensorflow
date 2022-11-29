import tensorflow as tf

print(tf.__version__)

tf.config.experimental.list_physical_devices(device_type=None)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


import numpy as np
import matplotlib.pyplot as plt
import keras as k

# from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras import backend as K
import time

# data preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_test = x_test.astype("float32")
x_train = x_train.astype("float32")
mean = np.mean(x_train)
std = np.std(x_train)
x_test = (x_test - mean) / std
x_train = (x_train - mean) / std

print(
    "counts of x_train : {}, y_train : {}, x_test : {}, y_test : {}".format(
        len(x_train), len(y_train), len(x_test), len(y_test)
    )
)

# labels
num_classes = 10
y_train = k.utils.to_categorical(y_train, num_classes)
y_test = k.utils.to_categorical(y_test, num_classes)
print(
    "counts of x_train : {}, y_train : {}, x_test : {}, y_test : {}".format(
        len(x_train), len(y_train), len(x_test), len(y_test)
    )
)

# CPU Train Model
num_filter = 32
num_dense = 512
drop_dense = 0.7
ac = "relu"
learningrate = 0.001

with tf.device("/cpu:0"):
    model = Sequential()

    model.add(
        Conv2D(
            num_filter, (3, 3), activation=ac, input_shape=(28, 28, 1), padding="same"
        )
    )
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(num_filter, (3, 3), activation=ac, padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 14x14x32

    model.add(Conv2D(2 * num_filter, (3, 3), activation=ac, padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(2 * num_filter, (3, 3), activation=ac, padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 7x7x64 = 3136 neurons

    model.add(Flatten())
    model.add(Dense(num_dense, activation=ac))
    model.add(BatchNormalization())
    model.add(Dropout(drop_dense))
    model.add(Dense(10, activation="softmax"))

    adm = Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)

cpu_list = []
batch_sizes = []
with tf.device("/cpu:0"):
    for i in range(0, 7):
        k = 8 * 2**i
        print("batch size " + str(k))
        t1 = time.time()
        model.fit(
            x_train, y_train, batch_size=k, epochs=1, validation_data=(x_test, y_test)
        )
        t2 = time.time()
        cpu_list.append(int(t2 - t1))
        batch_sizes.append(k)


# GPU Train Model
# build model

num_filter = 32
num_dense = 512
drop_dense = 0.7
ac = "relu"
learningrate = 0.001

model = Sequential()

model.add(
    Conv2D(num_filter, (3, 3), activation=ac, input_shape=(28, 28, 1), padding="same")
)
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(num_filter, (3, 3), activation=ac, padding="same"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 14x14x32

model.add(Conv2D(2 * num_filter, (3, 3), activation=ac, padding="same"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(2 * num_filter, (3, 3), activation=ac, padding="same"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))  # reduces to 7x7x64 = 3136 neurons

model.add(Flatten())
model.add(Dense(num_dense, activation=ac))
model.add(BatchNormalization())
model.add(Dropout(drop_dense))
model.add(Dense(10, activation="softmax"))

adm = Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=adm)

gpu_list = []
batch_sizes = []
print("gpu_list : ", gpu_list)
with tf.device("/gpu:0"):
    for i in range(0, 7):
        k = 8 * 2**i
        print("batch size " + str(k))
        t1 = time.time()
        model.fit(
            x_train, y_train, batch_size=k, epochs=1, validation_data=(x_test, y_test)
        )
        t2 = time.time()
        gpu_list.append(int(t2 - t1))
        batch_sizes.append(k)


# Summary Train Model results
# plot the comparison. The training with gpu is faster by a factor of about 4-6
plt.plot(batch_sizes, gpu_list, "bo")
plt.plot(batch_sizes, cpu_list, "ro")
plt.plot(batch_sizes, gpu_list, "b--")
plt.plot(batch_sizes, cpu_list, "r--")
plt.ylabel("training time per epoch (s)")
plt.xlabel("batch size")
plt.legend(["gpu", "cpu"], loc="upper right")
plt.ylim([0, 400])
# plt.savefig('CPUvsGPU.png')
plt.show()
