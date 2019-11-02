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

interaction_dat = pandas.read_csv('3-10-3-interaction-3.csv')
print(interaction_dat.head())
print(interaction_dat.describe())

sns.scatterplot(
    x='product',
    y='sales',
    hue='clerk',
    legend="full",
    data=interaction_dat
)
plt.show()

sales = interaction_dat['sales']
sample_num = len(sales)
product = interaction_dat['product']
clerk = interaction_dat['clerk']

stan_data = {
    'N': sample_num,
    'sales': sales,
    'product': product,
    'clerk': clerk
}

if os.path.exists('3-10-3-qua-qua.pkl'):
    sm = pickle.load(open('3-10-3-qua-qua.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-10-3-qua-qua.stan')

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

label_one = np.arange(1,10)
label_two = np.arange(10, 51)

y_clerks = []
for i in label_one:
    y = []
    for j in label_two:
        y.append((df['Intercept'].mean() + df['b_pro'].mean() * j + df['b_cl'].mean() * i + df[
            'b_pro_cl'].mean() * i * j))
    y_clerks.append(y)

pallet = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, y_clerk in enumerate(y_clerks):
    plt.plot(label_two, np.array(y_clerk), pallet[i], label='clerk %d' % (i+1))
plt.scatter(interaction_dat["product"],
            interaction_dat["sales"],
            c='k')
plt.legend()
plt.show()

# saving compiled model
if not os.path.exists('3-10-3-qua-qua.pkl'):
    with open('3-10-3-qua-qua.pkl', 'wb') as f:
        pickle.dump(sm, f)