import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


#Definición de datos y preprocesado y preparación de la Base de Datos
batchSize = 100
numClasses = 10
epochs = 10

rows,cols = 28,28 #Tamaño en píxeles de las imagenes en la base de datos mnist

(xt,yt),(xtest,ytest) = mnist.load_data() #Se cargan las imagenes en las variables de entrada salida y de validación o test

xt = xt.reshape(xt.shape[0], rows, cols, 1)
xtest = xtest.reshape(xtest.shape[0], rows, cols, 1)

xt = xt.astype('float32')
xtest = xtest.astype('float32')

xt = xt/255
xtest = xtest/255

yt = keras.utils.to_categorical(yt,numClasses)
ytest = keras.utils.to_categorical(ytest,numClasses)

#Creación y Compilación del modelo de la red neuronal
#RED NEURONAL CONVULACIONAL
#convModel = Sequential()
#convModel.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape=(28,28,1)))
#convModel.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
#convModel.add(MaxPool2D(pool_size = (2,2)))
#convModel.add(Flatten())
#convModel.add(Dense(68))
#convModel.add(Dropout(0.25))
#convModel.add(Dense(20))
#convModel.add(Dropout(0.25))
#convModel.add(Dense(num_classes,activation = 'softmax'))

#RED NEURONAL CLÁSICA
classicModel = Sequential()
classicModel.add(Flatten(input_shape = (28,28,1))) #Transformación de una imagen de 2 Dimensiones a 1 Dimensión
classicModel.add(Dense(68,activation = 'relu')) #Capa de neuronas de la Red Neuronal de 68 neuronas
#La activación con la función relu optimiza el entrenamiento y el aprendizaje en tiempo real
classicModel.add(Dropout(0.05)) #25% de probabilidades de que se anulen ciertas neuronas por época
classicModel.add(Dense(25,activation = 'relu')) #Otra capa de neuronas esta vez con 25
classicModel.add(Dropout(0.12)) #Se regulariza el valor de anulación en la segunda capa de neuronas
classicModel.add(Dense(numClasses,activation = 'softmax')) #La capa de salida siempre contendrá el numero de clases que hemos establecido anteriormente
#Softmax ayuda a repartir las probabilidades entre los componentes que participen en el entrenamiento
classicModel.summary() #Nos proporciona un resumen de la Red Neuronal antes de que se ejecute el entrenamiento


classicModel.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics = ['categorical_accuracy'])


#Entrenamiento y Evaluación de la red neuronal
classicModel.fit(x = xt, y = yt, batch_size = batchSize, epochs = epochs, validation_data = (xtest,ytest), verbose = 1)

punctuation = classicModel.evaluate(xtest, ytest, verbose = 1)

print(punctuation)