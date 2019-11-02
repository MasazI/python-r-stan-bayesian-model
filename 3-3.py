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
mcmc_result.plot()
plt.show()

mcmc_sample = mcmc_result.extract(permuted=True)

print(mcmc_sample['sales_pred'].shape)

az.plot_forest([mcmc_sample['sales_pred'].transpose(), mcmc_sample['mu_pred'].transpose()],
               model_names=['sales_pred', 'mu_pred'])
plt.show()

az.plot_forest([mcmc_sample['sales_pred'].transpose()[0], mcmc_sample['sales_pred'].transpose()[19]],
               model_names=['sales_pred 1', 'sales_pred 20'], kind='ridgeplot')
plt.show()



# saving compiled model
if not os.path.exists('3-3-1-simple-lm-pred.pkl'):
    with open('3-3-1-simple-lm-pred.pkl', 'wb') as f:
        pickle.dump(sm, f)

