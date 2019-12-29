import time
import pickle
import random
import numpy as np

import keras
from keras import Model
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10, fashion_mnist

from multinomial_class import HierarchicalModel, AllVsAllModel

np.random.seed(42)
random.seed(42)

dataset = "cifar10" # {"mnist", "cifar10", "fashion_mnist"}

# Load Data
if dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    labels = ['0','1','2','3','4','5','6','7','8','9']
elif dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
elif dataset == 'fashion_mnist':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
else:
    raise(ValueError(f"I don't know what {dataset} is!\nChange to one of 'mnist', 'cifar10', or 'fashion_mnist'."))

# Normalize and Reshape Data
x_train_orig = x_train.astype('float32') / 255.
x_test_orig = x_test.astype('float32') / 255.

if dataset in ['mnist','fashion_mnist']:
    x_train_orig = x_train_orig.reshape(x_train_orig.shape + (1,))
    x_test_orig = x_test_orig.reshape(x_test_orig.shape + (1,))

# Encode Labels
y_train_list = y_train.ravel()
y_test_list = y_test.ravel()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train_orig = x_train_orig[y_train_list < 5]
y_train = y_train[y_train_list < 5]
y_train_list = y_train_list[y_train_list < 5]
x_test_orig = x_test_orig[y_test_list < 5]
y_test = y_test[y_test_list < 5]
y_test_list = y_test_list[y_test_list < 5]

x_train = np.array(x_train_orig)[::10,:,:,:]
x_test = np.array(x_test_orig)[::1,:,:,:]
y_train = y_train[::10,:]
y_test = y_test[::1,:]
y_train_list = y_train_list[::10]
y_test_list = y_test_list[::1]

def build_LeNet(shape):
    l1 = keras.Input(shape=shape)
    l2 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(l1)
    l2 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(l2)
    l2 = keras.layers.MaxPooling2D(pool_size=(2,2))(l2)
    l2 = keras.layers.Flatten()(l2)
    l2 = keras.layers.Dropout(.25)(l2)
    l2 = keras.layers.Dense(units=256, activation='relu')(l2)
    l2 = keras.layers.Dropout(.5)(l2)
    l2 = keras.layers.Dense(units=10, activation='softmax')(l2)
    return (l1, l2)


trials = 1

shuffle = True
epochs = 12
epochs_sub = 20
batch_size = 128
batch_size_sub = 64

class_proportions = [100]

for prop in class_proportions:
    try:
        results_ova = pickle.load(open(f"results_ova_prop_{str(prop).zfill(3)}.pickle", "rb"))
    except:
        results_ova = []
    try:
        results_ava = pickle.load(open(f"results_ava_prop_{str(prop).zfill(3)}.pickle", "rb"))
    except:
        results_ava = []
    try:
        results_hie = pickle.load(open(f"results_hie_prop_{str(prop).zfill(3)}.pickle", "rb"))
    except:
        results_hie = []

    for _ in range(trials):
        keep_idx = []

        for i in range(len(set(y_train_list))):
            idx = list(np.where(y_train_list==i)[0].ravel())
            idx = random.sample(idx, int(np.ceil((prop/100.0)*len(idx))))
            keep_idx.extend(idx)

        x_train_sub = x_train[keep_idx]
        y_train_sub = y_train[keep_idx]
        y_train_list_sub = y_train_list[keep_idx]
        pickle.dump(results_test, open(f"results_test_prop_{str(prop).zfill(3)}.pickle", "wb"))

        tt = time.time()
        # One vs. All
        print("\nProcessing OvA\n")
        l1, l2 = build_LeNet(x_train.shape[1:])
        m = Model(l1, l2)
        m.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
        m.fit(x=x_train_sub,
              y=y_train_sub,
              shuffle=shuffle,
              epochs=epochs,
              batch_size=batch_size)
        results_ova.append(np.argmax(np.array(m.predict(x_test)), axis=-1)[:, None])
        pickle.dump(results_ova, open(f"results_ova_prop_{str(prop).zfill(3)}.pickle", "wb"))

        # All vs. All
        print("\nProcessing AvA\n")
        l1, l2 = build_LeNet(x_train.shape[1:])
        m = AllVsAllModel(l1, l2)
        m.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
        m.fit(x=x_train_sub,
              y=y_train_sub,
              shuffle=shuffle,
              epochs=epochs,
              epochs_sub=epochs_sub,
              batch_size=batch_size,
              batch_size_sub=batch_size_sub)
        results_ava.append(np.array(m.predict(x_test))[:, None])
        pickle.dump(results_ava, open(f"results_ava_prop_{str(prop).zfill(3)}.pickle", "wb"))

        # Hierarchical
        print("\nProcessing HIE\n")
        l1, l2 = build_LeNet(x_train.shape[1:])
        m = HierarchicalModel(l1, l2)
        m.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])
        m.fit(x=x_train_sub,
              y=y_train_sub,
              shuffle=shuffle,
              epochs=epochs,
              epochs_sub=epochs_sub,
              batch_size=batch_size,
              batch_size_sub=batch_size_sub)
        results_hie.append(np.array(m.predict(x_test))[:, None])
        pickle.dump(results_hie, open(f"results_hie_prop_{str(prop).zfill(3)}.pickle", "wb"))
        print(time.time() - tt)
