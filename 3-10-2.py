###############
#
# Transform R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
#
###############

import os

import numpy as np
import pystan
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

interaction_dat = pandas.read_csv('3-10-2-interaction-2.csv')
print(interaction_dat.head())
print(interaction_dat.describe())

sns.scatterplot(
    x='temperature',
    y='sales',
    hue='publicity',
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
temperature = interaction_dat_d['temperature']

stan_data = {
    'N': sample_num,
    'sales': sales,
    'publicity': publicity,
    'temperature': temperature
}

if os.path.exists('3-10-2-cat-qua.pkl'):
    sm = pickle.load(open('3-10-2-cat-qua.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-10-2-cat-qua.stan')

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

mcmc_sample = mcmc_result.extract()
df = pandas.DataFrame(mcmc_sample)
print(df.head())

label_one = ['not publicity', 'publicity']
label_two = np.arange(0,31)

y_p = []
for i in label_two:
    y_p.append((df['Intercept'].mean() + df['b_temp'].mean() * i + df['b_pub'].mean() * 1 + df['b_pub_temp'].mean() * 1 * i))

y_n = []
for i in label_two:
    y_n.append((df['Intercept'].mean() + df['b_temp'].mean() * i + df['b_pub'].mean() * 0 + df['b_pub_temp'].mean() * 0 * i))

plt.plot(label_two, np.array(y_p), 'red', label='publicity')
plt.scatter(interaction_dat_d.query('publicity_to_implement == 1')["temperature"],
            interaction_dat_d.query('publicity_to_implement == 1')["sales"],
            c='r')

plt.plot(label_two, np.array(y_n), 'blue', label='not publiciry')
plt.scatter(interaction_dat_d.query('publicity_not == 1')["temperature"],
            interaction_dat_d.query('publicity_not == 1')["sales"],
            c='b')

plt.legend()
plt.show()

# saving compiled model
if not os.path.exists('3-10-2-cat-qua.pkl'):
    with open('3-10-2-cat-qua.pkl', 'wb') as f:
        pickle.dump(sm, f)