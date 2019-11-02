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

file_beer_sales_ab = pandas.read_csv('2-6-1-beer-sales-ab.csv')
print(file_beer_sales_ab.head())

sns.distplot(file_beer_sales_ab.query("beer_name == 'A'")['sales'], bins=20, label='A')
sns.distplot(file_beer_sales_ab.query("beer_name == 'B'")['sales'], bins=20, label='B')
plt.legend()
plt.show()

sales_a = file_beer_sales_ab.query("beer_name == 'A'")['sales']
sales_b = file_beer_sales_ab.query("beer_name == 'B'")['sales']

sample_size = len(sales_b)
print(sample_size)

stan_data = {
    'N': 100,
    'sales_a': sales_a,
    'sales_b': sales_b
}

if os.path.exists('2-6-5-difference-mean.pkl'):
    sm = pickle.load(open('2-6-5-difference-mean.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='2-6-5-difference-mean.stan')

mcmc_result_6 = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

print(mcmc_result_6)

mcmc_samples = mcmc_result_6.extract()
az.plot_kde(mcmc_samples['diff'])
plt.show()

# saving complied model
if not os.path.exists('2-6-5-difference-mean.pkl'):
    with open('2-6-5-difference-mean.pkl', 'wb') as f:
        pickle.dump(sm, f)

