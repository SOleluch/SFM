# -*- coding: utf-8 -*-
"""LSTM_ALL (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZxPnN2JC_aqAT5nVRxVbaBb6FiBloAD7
"""

# importer les packages

import datetime as dt
import pandas as pd
import time
#from google.colab import drive
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input, LSTM, SimpleRNN, GaussianNoise
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

#drive.mount('drive', force_remount=True)

# Chargement des données 
#PATH = "/content/drive/MyDrive/GM5/SID/Deep Learning/Papier/"

date = pd.read_csv('data_hist_2007_2016.csv').loc[:,"date"]

data = pd.read_csv(f'data_qlib.csv')
data = data.iloc[:,1:]
data["date"] = date

print(data)

# Normalisation des données colonnes par colonnes

scalers = []

for c in data.columns:
  if c != "date":
    temp = data.loc[:,c].values
    temp = np.reshape(temp,(temp.shape[0],1))
    scaler = MinMaxScaler(feature_range=(-1,1))
    temp = scaler.fit_transform(temp)
    scalers.append(scaler)
    temp = np.reshape(temp,(temp.shape[0]))
    data.loc[:,c] = temp



print(data)

# Définition des paramètres

# prédiction pour le n-ième jour
n = 1

# Séparation des données en tenant compte de la jointure
data['date'] = pd.to_datetime(data['date'])
dataApp = data[data["date"] <= dt.datetime(2014,12,31)]
dataValid = data[data["date"] <= dt.datetime(2015,12,31)]
dataTest = data
dataApp = dataApp.to_numpy()
dataValid = dataValid.to_numpy()
dataTest = dataTest.to_numpy()


# On supprime la colonne date
dataApp = dataApp[:,:-1]
dataValid = dataValid[:,:-1]
dataTest = dataTest[:,:-1]

Napp,papp = dataApp.shape
Nvalid,pvalid = dataValid.shape
Ntest,ptest = dataTest.shape

dataApp = np.reshape(dataApp,(Napp,papp,1))
dataValid = np.reshape(dataValid,(Nvalid,pvalid,1))
dataTest = np.reshape(dataTest,(Ntest,ptest,1))

print(dataApp.shape)
print(dataValid.shape)

# Définition de nos échantillons X_train et X_test

X_train = dataApp[:Napp-n,:]
y_train = dataApp[n:,:]

X_test = dataTest[:Ntest-n,:]
y_test = dataTest[n:,:]

X_valid = dataValid[:Nvalid-n,:]
y_valid = dataValid[n:,:]

# On précise que l'on a que des types float et pas par exemple des nombres binaires
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_valid= np.asarray(X_valid).astype(np.float32)
y_valid = np.asarray(y_valid).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

val_len = X_valid.shape[0] - X_train.shape[0]
test_len = X_test.shape[0] - X_valid.shape[0]

print(val_len)
print(test_len)

# Définition paramètres du réseau
n_cells = 10
n_epochs = 2000
n_batch = 20
bruit = 0.001

# Construction du modèle 
model = Sequential()
#model.add(GaussianNoise(bruit))
model.add(LSTM(n_cells,return_sequences=True,input_shape=(papp,1)))#,kernel_initializer='random_uniform',recurrent_initializer='orthogonal',bias_initializer='zeros'))
#model.add(LSTM(n_cells,return_sequences=True))
#model.add(LSTM(n_cells,return_sequences=False))
#model.add(Dense(n_cells))
model.add(Dense(1))

model.build(X_train[0].shape)
print(X_train.shape)
model.summary()

# Compilation du modèle
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),loss="mean_squared_error")
#model.compile(optimizer="adam",loss="mean_squared_error")

TRAINING = False

model_name = f'Modèles/LSTM_ALL_derniers jours pour le {n}-ieme jour suivant_{n_cells}.hdf5'
  

if TRAINING == True : 

  checkpoint = ModelCheckpoint(filepath=model_name,monitor ='val_loss',verbose=1,save_best_only=True,mode='min')
  Enregistrement = model.fit(X_train, y_train, batch_size=n_batch,epochs=n_epochs,validation_data=(X_valid[-val_len:,:],y_valid[-val_len:,:]),callbacks=[checkpoint])

  print(Enregistrement.history.keys())

  plt.figure(figsize=(16,8))
  loss_train = Enregistrement.history['loss']
  val_loss_train = Enregistrement.history['val_loss']
  plt.plot(loss_train,"b:o", label = "train_loss")
  plt.plot(val_loss_train,"r:o", label = "val_loss")
  plt.title("Loss and over training epochs")
  plt.legend()
  plt.show()

model = keras.models.load_model(model_name)

# erreur dénormalisée
pred = model.predict(X_test)
pred = pred[-test_len:]
y_test = y_test[-test_len:]

for i in range(ptest):
  temp = y_test[:,i]
  temp = scalers[i].inverse_transform(temp)
  y_test[:,i] = temp
  temp = pred[:,i]
  temp = scalers[i].inverse_transform(np.reshape(temp,(len(temp),1)))
  #temp = np.reshape(temp,len(temp))
  pred[:,i] = temp


plt.plot(pred[:,0])
plt.plot(y_test[:,0])
plt.show()



  # RMSE
mse = np.mean(((pred[:,:,0] - y_test[:,:,0])**2))
print(f'\n\nOn obtient MSE={mse} sur X_test\n\n')

