###############
#
# Translate R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import os

import pystan
import pandas
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_beer_sales_1 = pandas.read_csv('2-4-1-beer-sales-1.csv')
print(file_beer_sales_1.head())

sample_size = len(file_beer_sales_1)

stan_data = {
    'N': sample_size,
    'sales':file_beer_sales_1['sales']
}

if os.path.exists('2-4-1-calc-mean-variance.pkl'):
    sm = pickle.load(open('2-4-1-calc-mean-variance.pkl', 'rb'))
else:
    sm = pystan.StanModel(file='2-4-1-calc-mean-variance.stan')

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

# saving complied model
with open('2-4-1-calc-mean-variance.pkl', 'wb') as f:
    pickle.dump(sm, f)

print(mcmc_result)

mcmc_result.plot()

# default viewer
plt.show()

# reference http://statmodeling.hatenablog.com/entry/pystan-rstanbook-chap5-1
ms = mcmc_result.extract(permuted=False, inc_warmup=True)
iter_from = mcmc_result.sim['warmup']
burn_range = np.arange(0, ms.shape[0])
iter_range = np.arange(iter_from, ms.shape[0])

paraname = mcmc_result.sim['fnames_oi']

print(paraname)

palette = sns.color_palette()
plt.figure()

# without burn in range
for pi in range(len(paraname)):
    plt.subplot(3, 1, pi+1)
    plt.tight_layout()
    [plt.plot(iter_range + 1, ms[iter_range,ci,pi], color=palette[ci]) for ci in range(ms.shape[1])]
    plt.title(paraname[pi])

plt.show()

# with burn in range
for pi in range(len(paraname)):
    plt.subplot(3, 1, pi+1)
    plt.tight_layout()
    [plt.plot(burn_range + 1, ms[burn_range,ci,pi], color=palette[ci]) for ci in range(ms.shape[1])]
    plt.title(paraname[pi])

plt.show()
