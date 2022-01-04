# -*- coding: utf-8 -*-
"""LSTM_ALL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r7Y7G1GIskFGyyZeE2wGuUq8OqCQC3-c
"""

# importer les packages

import datetime as dt
import pandas as pd
import time
#from google.colab import drive
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input, LSTM, SimpleRNN, GaussianNoise, GRU
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

#drive.mount('drive', force_remount=True)

# num_data = 1 si les notre 2 si qlib
num_data = 2

# Chargement des données 
#PATH = "/content/drive/MyDrive/GM5/SID/Deep Learning/Papier/"

date = pd.read_csv('data_hist_2007_2016.csv').loc[:,"date"]

if num_data == 1:
  df = pd.read_csv('data_hist_2007_2016.csv')

# Sélection seulement des cours "open"
  col = df.columns
  new_col = []

  for c in col :
    if "_open" in c:
      new_col.append(c)
    if "date" in c:
      new_col.append(c)

#new_col.append('date')
  data = pd.DataFrame(df, columns = new_col)

elif num_data == 2:
  data = pd.read_csv('data_qlib.csv')
  data = data.iloc[:,1:]
  data["date"] = date

print(data)

# choix = 1 on normalise action par action
# choix = 2 on normalise toutes les actions
choix = 1

# Normalisation des données colonnes par colonnes

if choix == 1:
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


elif choix == 2:

  min = data.iloc[:,:-1].min(skipna = True).min(skipna = True)
  max = data.iloc[:,:-1].max(skipna = True).max(skipna = True)

  scaler = MinMaxScaler(feature_range=(-1,1))
  scaler.fit_transform([[min],[max]])

  for c in data.columns:
    if c != "date":
      temp = data.loc[:,c].values
      temp = np.reshape(temp,(len(temp),1))
      temp = scaler.transform(temp)
      temp = np.reshape(temp,(len(temp)))
      data.loc[:,c] = temp


  print(data)

# Définition des paramètres

# w est le nombre de prix passsés impliqué pour prédire notre valeur future p_t+n
w = [3]#,5,10,15,20]
w_max = np.max(w)

# prédiction pour le n-ième jour
n = 1

# Séparation des données en tenant compte de la jointure
data['date'] = pd.to_datetime(data['date'])
dataApp = data[data["date"] <= dt.datetime(2014,12,31)]
dataValid = data[data["date"] <= dt.datetime(2015,12,31)]
dataValid = dataValid[dataValid["date"] > dt.datetime(2014,12,31)]
dataTest = data[data["date"] > dt.datetime(2015,12,31)]
dataApp = dataApp.to_numpy()
dataValid = dataValid.to_numpy()
dataTest = dataTest.to_numpy()
dataValid = np.concatenate((dataApp[-w_max:,:],dataValid),axis=0)
dataTest = np.concatenate((dataValid[-w_max:,:],dataTest),axis=0)


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
print(dataTest.shape)

print(dataApp[0:10,0])

# Définition de nos échantillons X_train et X_test

X_train = []
y_train = []

for j in range(papp):
  if j != 1 or num_data == 2:
    row = pd.DataFrame(dataApp[:,j,0]).first_valid_index()
    for i in range(row+w_max,Napp-n+1) :
      X_train.append(dataApp[i-w_max:i,j,0])
      y_train.append(dataApp[i+n-1,j,0])

X_test = []
y_test = []

for j in range(ptest):
  if j != 1 or num_data == 2:
    row = pd.DataFrame(dataTest[:,j,0]).first_valid_index()
    for i in range(row+w_max,Ntest-n+1) :
      X_test.append(dataTest[i-w_max:i,j,0])
      y_test.append(dataTest[i+n-1,j,0])

X_valid = []
y_valid = []

for j in range(pvalid):
  if j != 1 or num_data == 2:
    row = pd.DataFrame(dataValid[:,j,0]).first_valid_index()
    for i in range(row+w_max,Nvalid-n+1) :
      X_valid.append(dataValid[i-w_max:i,j,0])
      y_valid.append(dataValid[i+n-1,j,0])
    if row != 0:
      print(j)

# On précise que l'on a que des types float et pas par exemple des nombres binaires
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_valid= np.asarray(X_valid).astype(np.float32)
y_valid = np.asarray(y_valid).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# On mélange notre échantillon d'apprentissage
X_train, X_temp, y_train, y_temp = train_test_split(X_train,y_train,train_size=0.5, test_size=0.5, shuffle=True)
X_train = np.concatenate((X_train,X_temp))
y_train = np.concatenate((y_train,y_temp))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_valid = np.reshape(X_valid,(X_valid.shape[0],X_valid.shape[1],1))

# Définition paramètres du réseau
n_cells = 10
n_epochs = 200
n_batch = 50
bruit = 0.001

# Construction du modèle 
model = Sequential()
#model.add(GaussianNoise(bruit))
model.add(LSTM(n_cells,return_sequences=False,input_shape=(X_train.shape[1],X_train.shape[2])))#,kernel_initializer='random_uniform',recurrent_initializer='orthogonal',bias_initializer='zeros'))
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

for ww in w:

  model_name = f'Modèles/LSTM_ALL_{ww} derniers jours pour le {n}-ieme jour suivant_{n_cells}'
  
  if choix == 1:
    model_name = f'{model_name}.hdf5'
  elif choix == 2:
    model_name = f'{model_name}_ALLnorm.hdf5'
  

  if TRAINING == True : 

    checkpoint = ModelCheckpoint(filepath=model_name,monitor ='val_loss',verbose=1,save_best_only=True,mode='min')
    Enregistrement = model.fit(X_train, y_train, batch_size=n_batch,epochs=n_epochs,validation_data=(X_valid,y_valid),callbacks=[checkpoint])

    print(Enregistrement.history.keys())

    plt.figure(figsize=(16,8))
    loss_train = Enregistrement.history['loss']
    val_loss_train = Enregistrement.history['val_loss']
    plt.plot(loss_train,"b:o", label = "train_loss")
    plt.plot(val_loss_train,"r:o", label = "val_loss")
    plt.title("Loss and over training epochs")
    plt.legend()
    plt.show()

    #plt.figure(figsize=(30,15))
    #loss_train = Enregistrement.history['loss']
    #val_loss_train = Enregistrement.history['val_loss']
    #plt.plot(loss_train[15:],"b:o", label = "train_loss")
    #plt.plot(val_loss_train[15:],"r:o", label = "val_loss")
    #plt.title("Loss and over training epochs")
    #plt.legend()
    #plt.show()

  # Prédiction sur X_test
    pred_test = model.predict(X_test)

  # RMSE
    rmse = np.sqrt(np.mean(((pred_test[:,0] - y_test)**2)))
    mse = np.mean(((pred_test[:,0] - y_test)**2))
    print(f'\n\nOn obtient MSE={mse} sur X_test\n\n')

model = keras.models.load_model(model_name)

X = X_test
y = y_test

pred = model.predict(X)
pred = np.reshape(pred,(pred.shape[0]))

  # RMSE
rmse = np.sqrt(np.mean(((pred - y)**2)))
mse = np.mean(((pred- y)**2))
print(f'\n\nOn obtient MSE={mse} sur X_test\n\n')


X = X_train
y = y_train

pred = model.predict(X)
pred = np.reshape(pred,(pred.shape[0]))

  # RMSE
rmse = np.sqrt(np.mean(((pred - y)**2)))
mse = np.mean(((pred- y)**2))
print(f'\n\nOn obtient MSE={mse} sur X_train\n\n')


X = X_valid
y = y_valid

pred = model.predict(X)
pred = np.reshape(pred,(pred.shape[0]))


  # RMSE
rmse = np.sqrt(np.mean(((pred - y)**2)))
mse = np.mean(((pred- y)**2))
print(f'\n\nOn obtient MSE={mse} sur X_valid\n\n')

# erreur dénormalisée

pred = model.predict(X_test)
pred = np.reshape(pred,(pred.shape[0]))

if choix == 1:
  rank = Ntest-w_max-n+1
  for i in range(ptest):
    if i < 1 or num_data == 2:
      temp = y_test[i*rank:(i+1)*rank]
      temp = scalers[i].inverse_transform(np.reshape(temp,(len(temp),1)))
      temp = np.reshape(temp,len(temp))
      y_test[i*rank:(i+1)*rank] = temp
      temp = pred[i*rank:(i+1)*rank]
      temp = scalers[i].inverse_transform(np.reshape(temp,(len(temp),1)))
      temp = np.reshape(temp,len(temp))
      pred[i*rank:(i+1)*rank] = temp
    elif i > 1 and num_data == 1:
      temp = y_test[(i-1)*rank:i*rank]
      temp = scalers[i].inverse_transform(np.reshape(temp,(len(temp),1)))
      temp = np.reshape(temp,len(temp))
      y_test[(i-1)*rank:i*rank] = temp
      temp = pred[(i-1)*rank:i*rank]
      temp = scalers[i].inverse_transform(np.reshape(temp,(len(temp),1)))
      temp = np.reshape(temp,len(temp))
      pred[(i-1)*rank:i*rank] = temp

elif choix == 2:
  y_test = np.reshape(scaler.inverse_transform(np.reshape(y_test,(len(y_test),1))),(len(y_test)))
  pred = np.reshape(scaler.inverse_transform(np.reshape(pred,(len(pred),1))),(len(pred)))

  # RMSE
rmse = np.sqrt(np.mean(((pred - y_test)**2)))
mse = np.mean(((pred- y_test)**2))
print(f'\n\nOn obtient MSE={mse} sur X_test\n\n')

plt.plot(pred[0:100])
plt.plot(y_test[0:100])
plt.show()
