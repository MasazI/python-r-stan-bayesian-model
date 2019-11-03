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

fish_num_climate_3 = pandas.read_csv('4-2-1-fish-num-3.csv')
print(fish_num_climate_3.head())
print(fish_num_climate_3.describe())

sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='weather',
    data=fish_num_climate_3
)
plt.show()

sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='human',
    data=fish_num_climate_3
)
plt.show()

fish_num_climate_3_d = pandas.get_dummies(fish_num_climate_3, columns=["weather"])
print(fish_num_climate_3_d.head())

fish_num = fish_num_climate_3_d['fish_num']
sample_num = len(fish_num)
sunny = fish_num_climate_3_d['weather_sunny']
temperature = fish_num_climate_3_d['temperature']

# creating teamID
le = LabelEncoder()
le = le.fit(fish_num_climate_3['human'])
fish_num_climate_3['human'] = le.transform(fish_num_climate_3['human'])
sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='human',
    legend="full",
    data=fish_num_climate_3
)
plt.show()
human_id = fish_num_climate_3['human'].values
human_id = human_id + 1
human_num = len(np.unique(human_id))

print(human_id)

stan_data = {
    'N': sample_num,
    'fish_num': fish_num,
    'sunny': sunny,
    'temp': temperature,
    'N_human': human_num,
    'human_id': human_id
}


if os.path.exists('4-2-1-poisson-glmm.pkl'):
    sm = pickle.load(open('4-2-1-poisson-glmm.pkl', 'rb'))
    # sm = pystan.StanModel(file='4-2-1-poisson-glmm.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='4-2-1-poisson-glmm.stan')

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=6000,
    warmup=5000,
    thin=1
)

print(mcmc_result)
mcmc_result.plot()
plt.show()

# saving compiled model
if not os.path.exists('4-2-1-poisson-glmm.pkl'):
    with open('4-2-1-poisson-glmm.pkl', 'wb') as f:
        pickle.dump(sm, f)

mcmc_sample = mcmc_result.extract()
print(mcmc_sample)
df = pandas.DataFrame(mcmc_sample, columns=['Intercept', 'b_temp', 'b_sunny', 'sigma_r'])
r = mcmc_sample['r']
df_r = pandas.DataFrame(r)
print(df.head())
print(df_r.head())

label_one = ['cloudy', 'sunny']
label_two = np.arange(1,31)

for h in np.arange(human_num):
    y_s = []
    for i in label_two:
        y_s.append(np.exp(df['Intercept'].mean() + df['b_temp'].mean() * i + df['b_sunny'].mean() * 1 + df_r[h].mean()))

    y_c = []
    for i in label_two:
        y_c.append(np.exp(df['Intercept'].mean() + df['b_temp'].mean() * i + df['b_sunny'].mean() * 0 + df_r[h].mean()))

    plt.plot(label_two, np.array(y_s), 'red', label='sunny')
    plt.scatter(fish_num_climate_3.query('weather == "sunny" and human == %d' % h)["temperature"],
                fish_num_climate_3.query('weather == "sunny" and human == %d' % h)["fish_num"],
                c='r')

    plt.plot(label_two, np.array(y_c), 'blue', label='cloudy')
    plt.scatter(fish_num_climate_3.query('weather == "cloudy" and human == %d' % h)["temperature"],
                fish_num_climate_3.query('weather == "cloudy" and human == %d' % h)["fish_num"],
                c='b')

    plt.legend()
    plt.show()