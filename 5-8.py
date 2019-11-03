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

boat_df = pandas.read_csv('5-8-1-boat.csv')
print(boat_df.head())
print(boat_df.describe())
print(boat_df['x'])

# sales
plt.scatter(np.arange(0, len(boat_df['x'])), boat_df['x'], s=5)
plt.title("cambridge univ(1). vs oxford univ(0).")
plt.show()

x = boat_df['x']
T = len(x)
y = boat_df.dropna(how='any')['x'].values.astype(np.int32)
obs_no = boat_df.loc[boat_df.notnull()['x']].index.values + 1
len_obs = len(obs_no)

stan_data = {
    'T': T,
    'y': y,
    'obs_no': obs_no,
    'len_obs': len_obs
}

print("data length: %d" % T)
print("y length: %d" % len(y))
print("obs length: %d" % len_obs)

filename = '5-8-1-dglm-binom'

if os.path.exists('%s.pkl' % filename):
    sm = pickle.load(open('%s.pkl' % filename, 'rb'))
    # sm = pystan.StanModel(file='5-4-1-simple-reg.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='%s.stan' % filename)

control = {
    'adapt_delta': 0.8,
    'max_treedepth': 10
}

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=8000,
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
               np.arange(0, len(boat_df['x'])),
               'autoregressive local trend',
               'prob.',
               'probs',
               boat_df['x'])

