import numpy as np
import matplotlib.pyplot as plt
############################################################################(Array)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a[:2,1:3])
print(a[0][1])
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
'''Two ways of accessing the data in the middle row of the array. Mixing integer indexing with 
slices yields an array of lower rank, while using only slices yields an array of the 
same rank as the original array'''
row_r1=a[1,:]
row_r2=a[1:2,:]
row_r3=a[[1],:]
print(row_r1,row_r1.shape)
print(row_r2,row_r2.shape)
print(row_r3,row_r3.shape)
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1,col_r1.shape)
print(col_r2,col_r2.shape)
'''得到原数组的一部分，可以跳跃的选取'''
a = np.array([[1,2], [3, 4], [5, 6]])
# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])
# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx=(a>2)
print(bool_idx)
print(a[bool_idx])
print(a[a>2])
############################################################################(Array Math)
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print((x+y)==(np.add(x,y)))
print((x-y)==np.subtract(x,y))
print((x*y)==np.multiply(x,y))
print((x/y)==np.divide(x,y))
print(np.sqrt(x))
############################################################################(Broadcasting)
'''Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes 
when performing arithmetic operations. Frequently we have a smaller array and a larger array, 
and we want to use the smaller array multiple times to perform some operation on the larger 
array.
For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:'''
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y=np.empty_like(x)
for i in range(4):
    y[i,:]=x[i,:]+v
print(y)
vv=np.tile(v,(4,1))
print(vv==np.subtract(y,x))
#直接使用BroadCasting的例子
v=np.array([1,2,3])
w=np.array([4,5])
print(np.reshape(v,(3,1))*w)
x = np.array([[1,2,3], [4,5,6]])
print(x+v)

############################################################################(Matplotlib)
x=np.arange(0,3*np.pi,0.1)
y=np.sin(x)
plt.plot(x,y)
y_cos=np.cos(x)
plt.plot(x,y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine','Cosine'],loc='lower right')
plt.show()