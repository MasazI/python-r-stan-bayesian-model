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

sales_df_5 = pandas.read_csv('5-7-1-sales-ts-5.csv')
print(sales_df_5.head())
print(sales_df_5.describe())

# sales
sns.lineplot(x="date", y="sales", data=sales_df_5)
plt.show()

sales_df_5['date'] = pandas.to_datetime(sales_df_5['date'])
print(sales_df_5.head())
sales_df_5['date'] = sales_df_5['date'].view('int64') // 10 ** 9
print(sales_df_5.head())

sales = sales_df_5['sales']
T = len(sales)

stan_data = {
    'T': T,
    'y': sales
}

filename = '5-7-2-autoregressive'

if os.path.exists('%s.pkl' % filename):
    sm = pickle.load(open('%s.pkl' % filename, 'rb'))
    # sm = pystan.StanModel(file='5-4-1-simple-reg.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='%s.stan' % filename)

control = {
    'adapt_delta': 0.8,
    'max_treedepth': 16
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

# plot ssm of mu
tools.plot_ssm(mcmc_sample,
               sales_df_5['date'],
               'autoregressive local trend',
               'sales',
               'alpha',
               sales_df_5['sales'])
