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

sales_df_3 = pandas.read_csv('5-5-1-sales-ts-3.csv')
print(sales_df_3.head())
print(sales_df_3.describe())

sales_df_3['date'] = pandas.to_datetime(sales_df_3['date'])
print(sales_df_3.head())
sales_df_3['date'] = sales_df_3['date'].view('int64') // 10 ** 9
print(sales_df_3.head())

# sales
sns.lineplot(x="date", y="sales", data=sales_df_3)
plt.show()

sales = sales_df_3['sales']
T = len(sales)

stan_data = {
    'T': T,
    'sales': sales
}

filename = '5-5-2-smooth-trend'

if os.path.exists('%s.pkl' % filename):
    sm = pickle.load(open('%s.pkl' % filename, 'rb'))
    # sm = pystan.StanModel(file='5-4-1-simple-reg.stan')
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
    iter=8000,
    warmup=1200,
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
               sales_df_3['date'],
               'smooth trend model',
               'sales',
               'mu',
               sales_df_3['sales'])