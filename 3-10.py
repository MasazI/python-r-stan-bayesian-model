###############
#
# Transform R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
#
###############

import os

import pystan
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

interaction_dat = pandas.read_csv('3-10-1-interaction-1.csv')
print(interaction_dat.head())
print(interaction_dat.describe())

sns.scatterplot(
    x='publicity',
    y='sales',
    hue='bargen',
    data=interaction_dat
)
plt.show()

interaction_dat_d = pandas.get_dummies(interaction_dat)
print(interaction_dat_d.columns)
print(interaction_dat_d.head())
print(interaction_dat_d.describe())

sales = interaction_dat_d['sales']
sample_num = len(sales)
publicity = interaction_dat_d['publicity_to_implement']
bargen = interaction_dat_d['bargen_to_implement']

stan_data = {
    'N': sample_num,
    'sales': sales,
    'publicity': publicity,
    'bargen': bargen
}

if os.path.exists('3-10-1-cat-cat.pkl'):
    sm = pickle.load(open('3-10-1-cat-cat.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-10-1-cat-cat.stan')

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


# saving compiled model
if not os.path.exists('3-10-1-cat-cat.pkl'):
    with open('3-10-1-cat-cat.pkl', 'wb') as f:
        pickle.dump(sm, f)