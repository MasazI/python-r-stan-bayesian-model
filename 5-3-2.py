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
import tools

sales_df = pandas.read_csv('5-3-1-sales-ts-1-NA.csv')
print(sales_df.head())
print(sales_df.describe())

ax = sns.lineplot(x="date", y="sales", data=sales_df)
plt.show()

# drop NaN
sales_df_dropna = sales_df.dropna(how='any')
print(sales_df_dropna.head())
print(sales_df_dropna.describe())

print(len(sales_df['sales']))
print(len(sales_df_dropna['sales']))

# is Null
print(sales_df.isnull())

# getting not null index
print(sales_df.loc[sales_df.notnull()['sales']].index.values)

sales = sales_df['sales']
T = len(sales)
y = sales_df.dropna(how='any')['sales'].values
obs_no = sales_df.loc[sales_df.notnull()['sales']].index.values + 1
len_obs = len(obs_no)

print(len_obs)
print(obs_no)
print(y)

stan_data = {
    'T': T,
    'len_obs': len_obs,
    'y': y,
    'obs_no': obs_no
}

if os.path.exists('5-3-2-local-level-interpolation.pkl'):
    # sm = pickle.load(open('5-3-2-local-level-interpolation.pkl', 'rb'))
    sm = pystan.StanModel(file='5-3-2-local-level-interpolation.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='5-3-2-local-level-interpolation.stan')

control = {
    'adapt_delta': 0.8,
    'max_treedepth': 10
}

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    control=control,
    thin=1
)

print(mcmc_result)
mcmc_result.plot()
plt.show()

# saving compiled model
if not os.path.exists('5-3-2-local-level-interpolation.pkl'):
    with open('5-3-2-local-level-interpolation.pkl', 'wb') as f:
        pickle.dump(sm, f)

mcmc_sample = mcmc_result.extract()

# plot ssm
tools.plot_ssm(mcmc_sample,
               sales_df['date'],
               'local level model',
               'sales',
               'mu',
               sales_df['sales'])

# plot ssm about prediction
tools.plot_ssm(mcmc_sample,
               sales_df['date'],
               'local level model',
               'sales',
               'y_pred',
               sales_df['sales'])
