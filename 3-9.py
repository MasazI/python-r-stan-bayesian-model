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
from scipy.special import expit as logistic

germination_dat = pandas.read_csv('3-9-1-germination.csv')
print(germination_dat.head())
print(germination_dat.describe())

sns.scatterplot(
    x='nutrition',
    y='germination',
    hue='solar',
    data=germination_dat
)
plt.show()

germination_dat_d = pandas.get_dummies(germination_dat)

print(germination_dat_d.head())

sample_num = len(germination_dat_d['germination'])
germination = germination_dat_d['germination']
solar = germination_dat_d['solar_sunshine']
binom_size = germination_dat_d['size']
nutrition = germination_dat_d['nutrition']

stan_data = {
    'N': sample_num,
    'germination': germination,
    'solar': solar,
    'binom_size': binom_size,
    'nutrition': nutrition
}

if os.path.exists('3-9-1-glm-binom-1.pkl'):
    sm = pickle.load(open('3-9-1-glm-binom-1.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-9-1-glm-binom-1.stan')

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

label_one = ['shade', 'sunshine']
label_two = np.arange(1,11)

y_s = []
for i in label_two:
    y_s.append(logistic(df['Intercept'].mean() + df['b_nutrition'].mean() * i + df['b_solar'].mean() * 1)*10)

y_c = []
for i in label_two:
    y_c.append(logistic(df['Intercept'].mean() + df['b_nutrition'].mean() * i + df['b_solar'].mean() * 0)*10)


print(label_two.shape)
print(np.array(y_s).shape)
plt.plot(label_two, np.array(y_s), 'red', label='sunshine')
plt.scatter(germination_dat_d.query('solar_sunshine == 1')["nutrition"],
            germination_dat_d.query('solar_sunshine == 1')["germination"],
            c='r')

plt.plot(label_two, np.array(y_c), 'blue', label='shade')
plt.scatter(germination_dat_d.query('solar_shade == 1')["nutrition"],
            germination_dat_d.query('solar_shade == 1')["germination"],
            c='b')

plt.legend()
plt.show()

# saving compiled model
if not os.path.exists('3-9-1-glm-binom-1.pkl'):
    with open('3-9-1-glm-binom-1.pkl', 'wb') as f:
        pickle.dump(sm, f)

