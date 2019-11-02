###############
#
# Transform R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import os

import pystan
import pandas
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import arviz as az

animal_num = pandas.read_csv('2-5-1-animal-num.csv')
print(animal_num.head())

sample_size = len(animal_num)
print(sample_size)

stan_data = {
    'N': sample_size,
    'animal_num':animal_num['animal_num']
}

if os.path.exists('2-5-1-normal-dist.pkl'):
    sm_n = pickle.load(open('2-5-1-normal-dist.pkl', 'rb'))
else:
    sm_n = pystan.StanModel(file='2-5-1-normal-dist.stan')

mcmc_n_result = sm_n.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

# saving complied model
with open('2-5-1-normal-dist.pkl', 'wb') as f:
    pickle.dump(sm_n, f)


if os.path.exists('2-5-1-poisson-dist.pkl'):
    sm_p = pickle.load(open('2-5-1-poisson-dist.pkl', 'rb'))
else:
    sm_p = pystan.StanModel(file='2-5-1-poisson-dist.stan')

mcmc_p_result = sm_p.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

# saving complied model
with open('2-5-1-poisson-dist.pkl', 'wb') as f:
    pickle.dump(sm_p, f)

y_rep_normal = mcmc_n_result.extract()
y_rep_poisson = mcmc_p_result.extract()

# Posterior predictive shape
print(y_rep_normal['posterior_predictive'].shape)
print(y_rep_poisson['posterior_predictive'].shape)

# Posterior predictive plot (gaussian)
plt.figure()
plt.subplot(2, 3, 1)
plt.tight_layout()
plt.hist(animal_num['animal_num'], bins=20)

plt.subplot(2, 3, 2)
plt.tight_layout()
plt.hist(y_rep_normal['posterior_predictive'][0], bins=20)

plt.subplot(2, 3, 3)
plt.tight_layout()
plt.hist(y_rep_normal['posterior_predictive'][1], bins=20)

plt.subplot(2, 3, 4)
plt.tight_layout()
plt.hist(y_rep_normal['posterior_predictive'][2], bins=20)

plt.subplot(2, 3, 5)
plt.tight_layout()
plt.hist(y_rep_normal['posterior_predictive'][3], bins=20)

plt.subplot(2, 3, 6)
plt.tight_layout()
plt.hist(y_rep_normal['posterior_predictive'][4], bins=20)

plt.show()


# Posterior predictive plot (poisson)
plt.figure()
plt.subplot(2, 3, 1)
plt.tight_layout()
plt.hist(animal_num['animal_num'], bins=20)

plt.subplot(2, 3, 2)
plt.tight_layout()
plt.hist(y_rep_poisson['posterior_predictive'][0], bins=20)

plt.subplot(2, 3, 3)
plt.tight_layout()
plt.hist(y_rep_poisson['posterior_predictive'][1], bins=20)

plt.subplot(2, 3, 4)
plt.tight_layout()
plt.hist(y_rep_poisson['posterior_predictive'][2], bins=20)

plt.subplot(2, 3, 5)
plt.tight_layout()
plt.hist(y_rep_poisson['posterior_predictive'][3], bins=20)

plt.subplot(2, 3, 6)
plt.tight_layout()
plt.hist(y_rep_poisson['posterior_predictive'][4], bins=20)

plt.show()
