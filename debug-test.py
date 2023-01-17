import pandas


res_cl = pandas.read_excel(r'E:\residual chlorine\ResidualChlorine-test-debug.xlsx')
# res_cl = res_cl.drop(res_cl[res_cl.values[:, 1] == 0].index).dropna()
index_list = res_cl[res_cl.values[:, 0] == "val"].index
if len(index_list) == 1:
    i = index_list[0]
a = res_cl.iloc[:i, :]
b = res_cl.iloc[i+1:, :]
print()
