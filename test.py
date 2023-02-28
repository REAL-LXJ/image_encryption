from matplotlib import image
import numpy as np
# 下移shift_num1，右移shift_num2
def _circle_shift(array, shift_num1, shift_num2):
    h,w = array.shape
    array = np.vstack((array[(h-shift_num1):,:],array[:(h-shift_num2),:]))
    array = np.hstack((array[:,(w-shift_num2):],array[:,:(w-shift_num2)]))
    return array

def _array_shift(array, m, flag):
    array_new = np.roll(array, m, axis=flag)
    return array_new

def _array_test(array):
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            array[i,j] = array[i+1,j+1]
    print(array)

a = np.arange(1,26).reshape(5,5)
o = np.random.randint(0,2,(5,5))
#print(type(a))
#b = _circle_shift(a,3,3)
#b1 = _circle_shift(a,-3,-3)
#c = _array_shift(a,1,1)
#_array_test(a)
print("原矩阵A = \n", a)
print("密钥Key = \n", o)
#print("b = \n", b)
#print("b1 = \n", b1)
sum0 = np.sum(o == 0)
sum1 = np.sum(o == 1)
print("密钥中0的个数 = ", sum0)
print("密钥中1的个数 = ", sum1)
if sum0 > sum1:
    n = sum0 - sum1
    c = _circle_shift(a, n, n)
elif sum0 < sum1:
    n = sum1 - sum0
    c = _circle_shift(a, n, n)
else:
    c = a
print("依据密钥置乱的新矩阵C = \n", c)