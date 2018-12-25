# coding:utf-8
import pandas as pd
import numpy as np

file = '/Users/stern/Desktop/DT/Gold standard datasets/Drug_smi/Enzyme_Drug_Structure.xlsx'
df = pd.read_excel(file)
print('查看数据框前几行')
print df.head()

print('查看数据框的列数')
print(df.columns.size)

print('查看数据框的行数')
print(df.iloc[:, 0].size)

print('查看Series的行数')
print(len(df.iloc[:,0]))

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
print(type(d2))==pd.Series
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
#方法 iloc后产生的是Series



