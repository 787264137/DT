# coding:utf-8
import xlrd

wb = xlrd.open_workbook('/Users/stern/Desktop/Dataset/LargeScale/DictFile/Drug_smi/Enzyme_Drug_Structure.xlsx')

print wb.sheet_names()

sh = wb.sheet_by_index(0)

# 递归打印出每行的信息：
for rownum in range(sh.nrows):
    print sh.row_values(rownum)

# 只返回第一列数据：
first_column = sh.col_values(0)

print first_column
# 通过索引读取数据：
cell_A1 = sh.cell(0, 0).value

print cell_A1