###############
#
# Translate R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
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

file_beer_sales_1 = pandas.read_csv('2-4-1-beer-sales-1.csv')
print(file_beer_sales_1.head())

sample_size = len(file_beer_sales_1)

stan_data = {
    'N': sample_size,
    'sales':file_beer_sales_1['sales']
}

if not os.path.exists('2-4-1-calc-mean-variance.pkl'):
    print("Please execute 2-4.py in advance.")
    exit()

sm = pickle.load(open('2-4-1-calc-mean-variance.pkl', 'rb'))
mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

# reference http://statmodeling.hatenablog.com/entry/pystan-rstanbook-chap5-1
ms = mcmc_result.extract(permuted=False, inc_warmup=False)

print(type(ms))
print(ms.shape)

print(ms[0, 0, 0])
print(ms[:, 0, 0])
print(ms[:, 0, 0].shape)
print(ms[:, :, 0].shape)

mu_mcmc_vec = ms[:, :, 0].reshape(4000)
print(mu_mcmc_vec.shape)

# median
print(np.median(mu_mcmc_vec))

# mean
print(np.mean(mu_mcmc_vec))

print(np.quantile(mu_mcmc_vec, q=0.025))
print(np.quantile(mu_mcmc_vec, q=0.975))

iter_from = mcmc_result.sim['warmup']
iter_range = np.arange(0, ms.shape[0])

paraname = mcmc_result.sim['fnames_oi']

print(paraname)

palette = sns.color_palette()
plt.figure()

# traceplot
for pi in range(len(paraname)):
    plt.subplot(3, 1, pi+1)
    plt.tight_layout()
    [plt.plot(iter_range + 1, ms[iter_range,ci,pi], color=palette[ci]) for ci in range(ms.shape[1])]
    plt.title(paraname[pi])
plt.show()

#
for pi in range(len(paraname)):
    plt.subplot(3, 1, pi+1)
    plt.tight_layout()
    [sns.kdeplot(ms[iter_range,ci,pi], color=palette[ci]) for ci in range(ms.shape[1])]
    plt.title(paraname[pi])
plt.show()

mcmc_result.plot()

plt.show()

# using arviz instead of bayesplot
#az.plot_density(data=mcmc_result, var_names=['mu']);
az.plot_trace(data=mcmc_result)
plt.show()

az.plot_forest(data=mcmc_result, kind='ridgeplot', combined=True)
plt.show()

az.plot_autocorr(data=mcmc_result)
plt.show()

