import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# mnist_dataset= tf.keras.datasets.mnist
# # split into train and test
# # split into 80 20 
# # we will usesplit method in mnist, already split
# (x_train, y_train),(x_test,y_test)=mnist_dataset.load_data()
# # mnist_data.load_data() return two tuples for training and testing data
# # grayscale pixels have 0 to 255 lighting value
# # 0-255 => 0-1
# # NOTE:X is images, Y are labels(digits)
# # x_train =x_train/255.0
# # x_test =x_test/255.0
# x_train = tf.keras.utils.normalize(x_train,axis=1)
# x_test = tf.keras.utils.normalize(x_test,axis=1)
# model = tf.keras.models.Sequential()
# # now we created our nueral network
# # Sequential allows us to add layer above each others
# # Flatten layer:flat the input shape, i.e: a matrix of dimensions n*m became an array of size n*m
# # we used flatten to pass our image in the Dense layer of neral network
# # mnist inputs a matrix
# model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
# # 128 is the number of neurons, each of the 28*28 inputs will be connnected to each one of the 128 neuron
# model.add(tf.keras.layers.Dense(128,activation = 'relu'))
# # 'relu':rectify linear unit
# # f(x) = ramp(x) similar to ramp
# # for negative values, neurons will not learn
# # this activate the 128 neurons
# model.add(tf.keras.layers.Dense(128,activation = 'relu'))
# model.add(tf.keras.layers.Dense(10,activation = 'softmax'))
# # softmax gives the probability of each digit to be correct
# # the last Dense is the output layer
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
# model.fit(x_train,y_train,epochs=3)
# # to here, it is fully working model
# model.save('my.keras')
# to maintain same probability
model = tf.keras.models.load_model('my.keras')
# # loss,accuracy = model.evaluate(x_test,y_test)
# # print(loss,accuracy,end="\n")
# # those to check accuracy



imagenb=0
# while os.path.isfile(f"test{imagenb}.png"):

try:
    img = cv2.imread(f"test{imagenb}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The number {np.argmax(prediction)} appears")
    # argmax gixe the index of the field of highest probability
    # gives neuron with highest activation
    plt.imshow(img[0], cmap = plt.cm.binary)
    plt.show()
except: 
    print("Not clear!!")
# imagenb+=1
# for i in range(0,3):
#  print(os.path.isfile(f"test{i}.png"))












