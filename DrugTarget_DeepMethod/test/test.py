import numpy as np
x = np.asarray([[1,2,3],[2,3,4],[4,5,6]])
r,c = np.where(x>3) #标出二维矩阵根据条件筛选后的坐标
print(r)
print(c)

import tensorflow as tf
y = np.array([1,0,1,0])
y1 = tf.one_hot(y,depth=2,on_value=1)
y = np.asarray(y1)

sess = tf.Session()
with sess.as_default():
    print(y.eval())