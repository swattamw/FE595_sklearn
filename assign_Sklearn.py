import pandas as pd
import sklearn as sk


# Getting the Boston Housing dataset
bos_dat = sk.datasets.load_boston()
df_bos = pd.DataFrame(bos_dat.data)
df_bos.columns = bos_dat.feature_names

Price = pd.DataFrame(bos_dat.target)

# Fitting the Linear model

lm = sk.linear_model.LinearRegression()
lm.fit(df_bos[:], Price[0])



# Linear Regression Co-effients
coff = pd.Series(lm.coef_, index = bos_dat.feature_names)

coff = abs(coff)

coff.sort_values()



# Greatest and least influential variable

print('Most Influential -> ', coff.idxmax())

print('Least Influential -> ', coff.idxmin())
