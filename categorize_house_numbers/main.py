import numpy as np
import h5py 

hf = h5py.File(r'C:\Users\matt.moore\Downloads\test.tar\test\digitStruct.mat', 'r') 
data = np.array(hf['digitStruct/name'])
print(data.dtype)


foo = data[0][0]
print(hf[foo])
bar = [u''.join(chr(c) for c in hf[foo])]
print(bar)

'''
data = np.array(data) # For converting to numpy array
print(data.shape)
print(type(data[0,0]))
print(dir(data[0,0]))
'''
