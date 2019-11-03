###############
#
# Transform R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
#
###############

import os
import matplotlib.pyplot as plt
import pystan
import pandas
import pickle
import seaborn as sns
import numpy as np
import tools

fish_df = pandas.read_csv('5-9-1-fish-num-ts.csv')
print(fish_df.head())
print(fish_df.describe())
fish_df['date'] = pandas.to_datetime(fish_df['date'])
print(fish_df.head())
fish_df['date'] = fish_df['date'].view('int64') // 10 ** 9
print(fish_df.head())

y = fish_df['fish_num']
T = len(y)
ex = fish_df['temperature']

stan_data = {
    'T': T,
    'y': y,
    'ex': ex
}

filename = '5-9-1-dglm-poisson'

if os.path.exists('%s.pkl' % filename):
    # sm = pickle.load(open('%s.pkl' % filename, 'rb'))
    sm = pystan.StanModel(file='%s.stan' % filename)
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='%s.stan' % filename)

control = {
    'adapt_delta': 0.99,
    'max_treedepth': 16
}

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=3000,
    warmup=2000,
    control=control,
    thin=6
)

print(mcmc_result)
mcmc_result.plot()
plt.show()

# saving compiled model
if not os.path.exists('%s.pkl' % filename):
    with open('%s.pkl' % filename, 'wb') as f:
        pickle.dump(sm, f)

mcmc_sample = mcmc_result.extract(permuted=True)

# plot ssm of probs
tools.plot_ssm(mcmc_sample,
               fish_df['date'],
               'expectation of state',
               'prob.',
               'lambda_exp',
               fish_df['fish_num'])

tools.plot_ssm(mcmc_sample,
               fish_df['date'],
               'expectation of state without random',
               'prob.',
               'lambda_smooth',
               fish_df['fish_num'])

tools.plot_ssm(mcmc_sample,
               fish_df['date'],
               'expectation of state without random and fixed temperature',
               'prob.',
               'lambda_smooth_fix',
               fish_df['fish_num'])