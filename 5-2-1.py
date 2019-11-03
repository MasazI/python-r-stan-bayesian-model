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



sales_df = pandas.read_csv('5-2-1-sales-ts-1.csv')
print(sales_df.head())
print(sales_df.describe())

sales_df['date'] = pandas.to_datetime(sales_df['date'])
print(sales_df.head())
sales_df['date'] = sales_df['date'].view('int64') // 10**9
print(sales_df.head())

ax = sns.lineplot(x="date", y="sales", data=sales_df)
plt.show()

sales = sales_df['sales']
T = len(sales)

stan_data = {
    'T': T,
    'y': sales
}



if os.path.exists('5-2-1-local-level.pkl'):
    sm = pickle.load(open('5-2-1-local-level.pkl', 'rb'))
    # sm = pystan.StanModel(file='4-3-1-poisson-glmm.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='5-2-1-local-level.stan')

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
if not os.path.exists('5-2-1-local-level.pkl'):
    with open('5-2-1-local-level.pkl', 'wb') as f:
        pickle.dump(sm, f)


mcmc_sample = mcmc_result.extract()

# plot ssm
tools.plot_ssm(mcmc_sample,
               sales_df['date'],
               'local level model',
               'sales',
               'mu',
               sales_df['sales'])