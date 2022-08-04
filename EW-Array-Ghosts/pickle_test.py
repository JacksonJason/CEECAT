import pickle
import numpy as np

file = open("test.p", 'wb')

arr_1 = np.array([1,2,3])
arr_2 = np.array([4,5,6])

pickle.dump(arr_1, file)
pickle.dump(arr_2, file)

file = open("test.p", "rb")
arr_2 = pickle.load(file)
arr_1 = pickle.load(file)

print(arr_1, arr_2)