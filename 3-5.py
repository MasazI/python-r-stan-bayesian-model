###############
#
# Transform R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import os

import numpy as np
import pystan
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import arviz as az

file_beer_sales_2 = pandas.read_csv('3-2-1-beer-sales-2.csv')
print(file_beer_sales_2.head())

# sns.pairplot(file_beer_sales_2)
# plt.show()

print(len(file_beer_sales_2))

sample_num = len(file_beer_sales_2)

temperature_pred = range(11,31)

stan_data = {
    'N': sample_num,
    'sales': file_beer_sales_2['sales'],
    'temperature': file_beer_sales_2['temperature'],

    'N_pred': len(temperature_pred),
    'temperature_pred': temperature_pred
}

if os.path.exists('3-3-1-simple-lm-pred.pkl'):
    sm = pickle.load(open('3-3-1-simple-lm-pred.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-3-1-simple-lm-pred.stan')

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

print(mcmc_result)
# mcmc_result.plot()
# plt.show()

mcmc_sample = mcmc_result.extract(permuted=True)

print(mcmc_sample['sales_pred'].shape)

az.plot_forest([mcmc_sample['beta'], mcmc_sample['Intercept']])
plt.show()

# visualization of regression line
df = pandas.DataFrame(mcmc_sample['sales_pred'])
col = np.arange(11,31)
df.columns = col

qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()

for i in np.arange(len(df.columns)):
    for qu in qua:
        d_est[qu] = df.quantile(qu)


print(d_est)

x = d_est.index
y1 = d_est[0.025].values
y2 = d_est[0.25].values
y3 = d_est[0.5].values
y4 = d_est[0.75].values
y5 = d_est[0.975].values

plt.fill_between(x,y1,y5,facecolor='blue',alpha=0.1)
plt.fill_between(x,y2,y4,facecolor='blue',alpha=0.5)
plt.plot(x,y3,'k-')
plt.scatter(file_beer_sales_2["temperature"],file_beer_sales_2["sales"],c='b')

plt.show()

# saving compiled model
if not os.path.exists('3-3-1-simple-lm-pred.pkl'):
    with open('3-3-1-simple-lm-pred.pkl', 'wb') as f:
        pickle.dump(sm, f)

