import numpy as np

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
row_1 = a[1,:]
row_2 = a[1:2,:]
# print row_1,row_1.shape
# print row_2,row_2.shape

col_1 = a[:,1]
col_2 = a[:,1:2]
#
# print col_1,col_1.shape
# print col_2,col_2.shape

print a[[0,2,1,1]]