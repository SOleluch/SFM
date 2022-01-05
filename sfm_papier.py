# -*- coding: utf-8 -*-
"""LSTM_ALL (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZxPnN2JC_aqAT5nVRxVbaBb6FiBloAD7
"""

# importer les packages

from __future__ import absolute_import
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import theano.tensor as T
from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers.recurrent import Recurrent
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

class ITOSFM(Recurrent):

    def __init__(self, output_dim, freq_dim, hidden_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(ITOSFM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim
        
        self.states = [None, None, None, None, None]
        self.W_i = self.init((input_dim, self.hidden_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.hidden_dim,), name='{}_b_i'.format(self.name))

        self.W_ste = self.init((input_dim, self.hidden_dim),
                             name='{}_W_ste'.format(self.name))
        self.U_ste = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_ste'.format(self.name))
        self.b_ste = self.forget_bias_init((self.hidden_dim,),
                                         name='{}_b_ste'.format(self.name))

        self.W_fre = self.init((input_dim, self.freq_dim),
                             name='{}_W_fre'.format(self.name))
        self.U_fre = self.inner_init((self.hidden_dim, self.freq_dim),
                                   name='{}_U_fre'.format(self.name))
        self.b_fre = self.forget_bias_init((self.freq_dim,),
                                         name='{}_b_fre'.format(self.name))
        
        self.W_c = self.init((input_dim, self.hidden_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.hidden_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.hidden_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.hidden_dim,), name='{}_b_o'.format(self.name))
		
        self.U_a = self.inner_init((self.freq_dim, 1),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.hidden_dim,), name='{}_b_a'.format(self.name))
        
        self.W_p = self.init((self.hidden_dim, self.output_dim),
                             name='{}_W_p'.format(self.name))
        self.b_p = K.zeros((self.output_dim,), name='{}_b_p'.format(self.name))
        
        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_ste, self.U_ste, self.b_ste,
                                  self.W_fre, self.U_fre, self.b_fre,
                                  self.W_o, self.U_o, self.b_o,
                                  self.U_a, self.b_a,
                                  self.W_p, self.b_p]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        

    def get_initial_states(self, x):

        init_state_h = K.zeros_like(x)
        init_state_h = K.sum(init_state_h, axis = 1)
        reducer_s = K.zeros((self.input_dim, self.hidden_dim))
        reducer_f = K.zeros((self.hidden_dim, self.freq_dim))
        reducer_p = K.zeros((self.hidden_dim, self.output_dim))
        init_state_h = K.dot(init_state_h, reducer_s)
        
        init_state_p = K.dot(init_state_h, reducer_p)
        
        init_state = K.zeros_like(init_state_h)
        init_freq = K.dot(init_state_h, reducer_f)

        init_state = K.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = K.reshape(init_freq, (-1, 1, self.freq_dim))
        
        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq
        
        init_state_time = K.cast_to_floatx(0.)
        
        initial_states = [init_state_p, init_state_h, init_state_S_re, init_state_S_im, init_state_time]
        return initial_states

    def step(self, x, states):
        p_tm1 = states[0]
        h_tm1 = states[1]
        S_re_tm1 = states[2]
        S_im_tm1 = states[3]
        time_tm1 = states[4]
        B_U = states[5]
        B_W = states[6]
        frequency = states[7]
        
        x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
        x_ste = K.dot(x * B_W[0], self.W_ste) + self.b_ste
        x_fre = K.dot(x * B_W[0], self.W_fre) + self.b_fre
        x_c = K.dot(x * B_W[0], self.W_c) + self.b_c
        x_o = K.dot(x * B_W[0], self.W_o) + self.b_o
        
        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        
        ste = self.inner_activation(x_ste + K.dot(h_tm1 * B_U[0], self.U_ste))
        fre = self.inner_activation(x_fre + K.dot(h_tm1 * B_U[0], self.U_fre))

        ste = K.reshape(ste, (-1, self.hidden_dim, 1))
        fre = K.reshape(fre, (-1, 1, self.freq_dim))
        f = ste * fre
        
        c = i * self.activation(x_c + K.dot(h_tm1 * B_U[0], self.U_c))
        
        time = time_tm1 + 1

        omega = K.cast_to_floatx(2*np.pi)* time * frequency
        re = T.cos(omega)
        im = T.sin(omega)
        
        c = K.reshape(c, (-1, self.hidden_dim, 1))
        
        S_re = f * S_re_tm1 + c * re
        S_im = f * S_im_tm1 + c * im
        
        A = K.square(S_re) + K.square(S_im)

        A = K.reshape(A, (-1, self.freq_dim))
        A_a = K.dot(A * B_U[0], self.U_a)
        A_a = K.reshape(A_a, (-1, self.hidden_dim))
        a = self.activation(A_a + self.b_a)
        
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[0], self.U_o))

        h = o * a
        p = K.dot(h, self.W_p) + self.b_p

        return p, [p, h, S_re, S_im, time]

    def get_constants(self, x):
        constants = []
        constants.append([K.cast_to_floatx(1.) for _ in range(6)])
        constants.append([K.cast_to_floatx(1.) for _ in range(7)])
        array = np.array([float(ii)/self.freq_dim for ii in range(self.freq_dim)])
        constants.append([K.cast_to_floatx(array)])
        
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "fre_dim": self.fre_dim,
                  "hidden_dim": self.hidden_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(ITOSFM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Chargement des données 

date = pd.read_csv('data_hist_2007_2016.csv').loc[:,"date"]
data = pd.read_csv('data_qlib.csv')
data = data.iloc[:,1:]
data["date"] = date

#print(data)

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



#print(data)

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

dataApp = np.transpose(dataApp)
dataTest = np.transpose(dataTest)
dataValid = np.transpose(dataValid)

papp,Napp = dataApp.shape
pvalid,Nvalid = dataValid.shape
ptest,Ntest = dataTest.shape

dataApp = np.reshape(dataApp,(papp,Napp,1))
dataValid = np.reshape(dataValid,(pvalid,Nvalid,1))
dataTest = np.reshape(dataTest,(ptest,Ntest,1))

#print(dataApp.shape)
#print(dataValid.shape)

# Définition de nos échantillons X_train et X_test

X_train = dataApp[:,:Napp-n]
y_train = dataApp[:,n:]

X_test = dataTest[:,:Ntest-n]
y_test = dataTest[:,n:]

X_valid = dataValid[:,:Nvalid-n]
y_valid = dataValid[:,n:]


# On précise que l'on a que des types float et pas par exemple des nombres binaires
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_valid= np.asarray(X_valid).astype(np.float32)
y_valid = np.asarray(y_valid).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

val_len = X_valid.shape[1] - X_train.shape[1]
test_len = X_test.shape[1] - X_valid.shape[1]

#print(val_len)
#print(test_len)

# Définition paramètres du réseau
hidden_dim=50 
freq_dim=10
n_epochs = 4000
n_batch = 800
learning_rate=0.01

#build the model
def build_model(layers, freq, learning_rate):
    model = Sequential()

    model.add(ITOSFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        freq_dim = freq,
        return_sequences=True))

    
    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    return model

# Construction du modèle 
model = build_model([1, hidden_dim, 1], freq_dim, learning_rate)

model.summary()

TRAINING = False

model_name = 'Modèles/SFM_ALL_Pour le '+str(n)+'-ieme jour suivant_'+str(hidden_dim)+'_'+str(freq_dim)+'.hdf5'

model.load_weights(model_name)

if TRAINING == True : 

  checkpoint = ModelCheckpoint(filepath=model_name,monitor ='val_loss',verbose=1,save_best_only=True,mode='min')
  Enregistrement = model.fit(X_train, y_train, batch_size=n_batch,nb_epoch=n_epochs,validation_data=(X_valid[:,-val_len:],y_valid[:,-val_len:]),callbacks=[checkpoint])
  #model.save_weights(model_name, overwrite = True)

  print(Enregistrement.history.keys())

  plt.figure(figsize=(16,8))
  loss_train = Enregistrement.history['loss']
  val_loss_train = Enregistrement.history['val_loss']
  plt.plot(loss_train,"b:o", label = "train_loss")
  plt.plot(val_loss_train,"r:o", label = "val_loss")
  plt.title("Loss and over training epochs")
  plt.legend()
  plt.show()

if TRAINING == False :
    model = build_model([1, hidden_dim, 1], freq_dim, learning_rate)
    model.load_weights(model_name)

# erreur dénormalisée
    pred = model.predict(X_test)
    pred = pred[:,-test_len:]
    y_test = y_test[:,-test_len:]

    for i in range(ptest):
        temp = y_test[i,:]
        temp = scalers[i].inverse_transform(temp)
        y_test[i,:] = temp
        temp = pred[i,:]
        temp = scalers[i].inverse_transform(temp)
        pred[i,:] = temp

    plt.plot(pred[0,:])
    plt.plot(y_test[0,:])
    plt.show()

    # RMSE
    mse = np.mean(((pred - y_test)**2))
    print('\n\nOn obtient MSE='+str(mse)+' sur X_test\n\n')

