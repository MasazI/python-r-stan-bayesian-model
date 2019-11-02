###############
#
# Transform R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import os

import pystan
import pandas
import pickle

file_beer_sales_1 = pandas.read_csv('2-4-1-beer-sales-1.csv')
print(file_beer_sales_1.head())

sample_size = len(file_beer_sales_1)

stan_data = {
    'N': sample_size,
    'sales':file_beer_sales_1['sales']
}

if os.path.exists('2-6-1-calc-normal-prior.pkl'):
    sm = pickle.load(open('2-6-1-calc-normal-prior.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='2-6-1-calc-normal-prior.stan')

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

# saving complied model
with open('2-6-1-calc-normal-prior.pkl', 'wb') as f:
    pickle.dump(sm, f)

print(mcmc_result)