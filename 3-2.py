###############
#
# Transform R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import os

import pystan
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import arviz as az

file_beer_sales_2 = pandas.read_csv('3-2-1-beer-sales-2.csv')
print(file_beer_sales_2.head())

sns.pairplot(file_beer_sales_2)
plt.show()

print(len(file_beer_sales_2))

sample_num = len(file_beer_sales_2)

stan_data = {
    'N': sample_num,
    'sales': file_beer_sales_2['sales'],
    'temperature': file_beer_sales_2['temperature']
}

if os.path.exists('3-2-1-simple-lm.pkl'):
    sm = pickle.load(open('3-2-1-simple-lm.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-2-1-simple-lm.stan')

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

print(mcmc_result)

mcmc_sample = mcmc_result.extract(permuted=True)

print(mcmc_sample)

az.plot_trace(data=mcmc_sample, var_names=['Intercept', 'beta', 'sigma'], combined=False)
plt.show()

# saving compiled model
if not os.path.exists('3-2-1-simple-lm.pkl'):
    with open('3-2-1-simple-lm.pkl', 'wb') as f:
        pickle.dump(sm, f)
