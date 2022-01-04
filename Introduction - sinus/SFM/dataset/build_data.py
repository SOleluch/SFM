import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

directory = 'sinus'
filenames = os.listdir(directory)
print(filenames)

all_data = np.zeros((len(filenames),2000),dtype = np.float32)
for i in range(len(filenames)):
  filename = filenames[i]
  print(i)
  print(filename)
  
  data=pd.read_csv(directory+'/'+filename)
  vars = ['sinus']
  data = data[vars]
  data = np.array(data)
  data = np.transpose(data)
  data = data[0]
  
  all_data[i] = data

print(all_data.shape)  
np.save('data',all_data)

