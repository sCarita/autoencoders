"""
Simple MLP Autoencoder
"""
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print(X_train.shape)
print(X_test.shape)

# this is the size of our encoded representation
# 32 floats -> compression of factor 24.5, assuming the input is 784 floats
encoding_dim = 32

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" us the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps and input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retireve the laste layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# lets fit the model to the data
autoencoder.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# encode and decode some digits
# note that we take them from "test" set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

# How many digits we will display
n = 10
plt.figure(figsize=(20,4))

for i in range(n):
  # display original
  ax = plt.subplot(2, n, i+1)
  plt.imshow(X_test[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i+1+n)
  plt.imshow(decoded_imgs[i].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)


plt.show()
