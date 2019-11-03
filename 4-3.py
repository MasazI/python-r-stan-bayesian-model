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
from sklearn.preprocessing import LabelEncoder

fish_num_climate_4 = pandas.read_csv('4-3-1-fish-num-4.csv')
print(fish_num_climate_4.head())
print(fish_num_climate_4.describe())

sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='human',
    data=fish_num_climate_4
)
plt.show()

fish_num_climate_4_d = pandas.get_dummies(fish_num_climate_4, columns=["human"])
print(fish_num_climate_4_d.head())

fish_num = fish_num_climate_4_d['fish_num']
sample_num = len(fish_num)
temperature = fish_num_climate_4_d['temperature']

# creating teamID
le = LabelEncoder()
le = le.fit(fish_num_climate_4['human'])
fish_num_climate_4['human'] = le.transform(fish_num_climate_4['human'])

sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='human',
    legend="full",
    data=fish_num_climate_4
)
plt.show()

human_id = fish_num_climate_4['human'].values
human_num = len(np.unique(human_id))
human_id = (human_id + 1)
print(human_id)

stan_data = {
    'N': sample_num,
    'fish_num': fish_num,
    'temp': temperature,
    'human_id': human_id
}

if os.path.exists('4-3-1-poisson-glmm.pkl'):
    sm = pickle.load(open('4-3-1-poisson-glmm.pkl', 'rb'))
    # sm = pystan.StanModel(file='4-3-1-poisson-glmm.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='4-3-1-poisson-glmm.stan')

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

# saving compiled model
if not os.path.exists('4-3-1-poisson-glmm.pkl'):
    with open('4-3-1-poisson-glmm.pkl', 'wb') as f:
        pickle.dump(sm, f)

mcmc_sample = mcmc_result.extract()
print(mcmc_sample)

# visualization
label_temp = np.arange(10,21)
df = pandas.DataFrame(mcmc_sample, columns=['Intercept', 'b_temp', 'b_human', 'b_temp_human'])

for h in np.arange(human_num):
    y = []
    for i in label_temp:
        y.append(np.exp(df['Intercept'].mean() +
                        df['b_temp'].mean() * i +
                        df['b_human'].mean() * (h+1) +
                        df['b_temp_human'].mean() * (h+1) * i))

    plt.plot(label_temp, np.array(y), 'red', label='%d' % (h + 1))
    plt.scatter(fish_num_climate_4.query('human == %d' % h)["temperature"],
                fish_num_climate_4.query('human == %d' % h)["fish_num"],
                c='r')
    plt.legend()
    plt.show()

