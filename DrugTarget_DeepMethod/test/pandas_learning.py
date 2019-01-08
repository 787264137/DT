# coding:utf-8
import pandas as pd
import numpy as np

file = '/Users/stern/Desktop/DT/Gold standard datasets/Drug_smi/Enzyme_Drug_Structure.xlsx'
df = pd.read_excel(file)
print('查看数据框前几行')
print(df.head())

print('查看数据框的列数')
print(df.columns.size)

print('查看数据框的行数')
print(df.iloc[:, 0].size)

print('查看Series的行数')
print(len(df.iloc[:, 0]))

df1 = df.iloc[:, 0:1]
print('获取数据框第0列')
print(df1.head())

df2 = df.iloc[:, 1:2]
print('获取数据框第1列')
print(df2.head())

print('获取数据框第2行，第1列的值')
print(df.values[1][0])

print('对于第2行，第1列赋值为AAA,BBB')
df.iloc[1, 0] = 'AAA'
print(df.iloc[1, 0])
df.loc[1, "Drug"] = 'BBB'
print(df.iloc[1, 0])

print('列表转数据框,以行标准写入和以列标准写入或者通过np.array')
a = [[1, 2, 3, 4], [5, 6, 7, 8]]
data = pd.DataFrame(a)
print(data)
a1 = [1, 2, 3, 4]
b1 = [5, 6, 7, 8]
c1 = {"a": a1,
      "b": b1}
data = pd.DataFrame(c1)
print(data)
# 最好用的方法
data = pd.DataFrame(np.array(a))
print(data)

print('数据框转列表')
data_lst = np.array(data).tolist()
print(data_lst)

print("获取数据框的行索引，列索引")
print(list(np.array(data.index)))
print(list(np.array(data.columns)))

print('合并数据框')
# 报错，不行，两个数据框必须要有相同的一列
# data1 = pd.DataFrame(np.array([[11,12,13,14],[15,16,17,18]]),columns=['a','b','c','d'])
# print(data)
# print(data1)
# data_merge = pd.merge(data,data1)
# print(data_merge)
d1 = data.iloc[:, 0]
d2 = data.iloc[:, 3]
d3 = data.iloc[:, 2]
print(type(d1))
print(type(d2) == pd.Series)
print(d2[0])
d_merge = pd.DataFrame({'column1': d1, 'column2': d2, 'column3': d3})
# 注意，使用字典合并时，value必须是series，不然报错ValueError: If using all scalar values, you must pass an index
print(d_merge)

print('Series转DataFrame')
a = pd.Series(data=[1, 2, 3, 4])
print(type(a))
print(type(a.to_frame()))
print(type(a.tolist()))

print('DataFrame转Series')
# 方法 iloc后产生的是Series

print('数据框拼接')
# 数据框拼接（ignore_index=True,重新分配索引）
# 两种方式，concat、append皆可以
a = [[1, 2, 3, 4], [5, 6, 7, 8]]
result1 = pd.DataFrame(a)
c = [[11, 12, 13, 14], [5, 6, 7, 8]]
result2 = pd.DataFrame(c)
result3 = pd.concat([result1, result2], ignore_index=True)
result4 = result1.append(result2, ignore_index=True)
print('result3')
print(result3)
print('result4')
print(result4)
# 用法
# pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
#           keys=None, levels=None, names=None, verify_integrity=False,
#           copy=True)
#
# axis：要粘在哪个轴上。默认0，粘贴行。
# join：默认outer，合集；inner，交集。
# ignore_index：布尔型，默认False。如果为Ture的话，会重新分配index从0...n-1。
# keys：一个序列，默认None。建立等级索引，作为最外层的level。
# levels：序列sequences构成的list，默认None。
#
print("数据框合并会删除一些重复的")
# 数据框合并
a = [[1, 2, 3, 4], [5, 6, 7, 8]]
data2 = pd.DataFrame(a)
c = [[11, 12, 13, 14], [5, 6, 7, 8]]
data3 = pd.DataFrame(c)
data4 = pd.merge(data2, data3, how='outer')
print('data4')
print(data4)
# 用法
# pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
#          left_index=False, right_index=False, sort=True,
#          suffixes=('_x', '_y'), copy=True, indicator=False)
#
# left: 一个dataframe对象
# right: 另一个dataframe对象
# how: 可以是
# 'left', 'right', 'outer', 'inner'.默认为inner。
# on: 列名，两个dataframe都有的列。如果不传参数，
# 而且left_index和right_index也等于False，
# 则默认把两者交叉 / 共有的列作为链接键（join
# keys）。
# 可以是一个列名，也可以是包含多个列名的list。
# left_on: 左边dataframe的列会用做keys。可以是列名，
# 或者与dataframe长度相同的矩阵array。
# right_on: 右边同上。
# left_index: 如果为Ture，用左侧dataframe的index作为
# 连接键。如果是多维索引，level数要跟右边相同才行。
# right_index: 右边同上。
# sort: 对合并后的数据框排序，以连接键。
# suffixes: 一个tuple，包字符串后缀，用来加在重叠的列名后面。
# 默认是('_x', '_y')。
# copy: 默认Ture，复制数据。
# indicator: 布尔型（True / FALSE），或是字符串。
# 如果为True，合并之后会增加一列叫做
# '_merge'。
# 是分类数据，用left_only, right_only, both来标记
# 来自左边，右边和两边的数据。

print("数据框删除行")
a = [[1, 2, 3, 4], [5, 6, 7, 8]]
data2 = pd.DataFrame(a)
c = [[11, 12, 13, 14], [5, 6, 7, 8]]
data3 = pd.DataFrame(c)
result3 = pd.concat([result1, result2], ignore_index=True)
result3.drop(0, axis=0)

print("删除含有5的行")
rowNum = result3.iloc[:,0] == 5
for i in range(len(rowNum)):
      if rowNum[i]:
            result3 = result3.drop(i,axis=0)

print("删除空缺行")
df = df.dropna(how='any')