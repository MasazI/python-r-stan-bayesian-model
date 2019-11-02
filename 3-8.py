###############
#
# Transform R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import os

import numpy as np
import pystan
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import arviz as az

fish_num_climate = pandas.read_csv('3-8-1-fish-num-1.csv')
print(fish_num_climate.head())

sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='weather',
    data=fish_num_climate
)
plt.show()

fish_num_climate_d = pandas.get_dummies(fish_num_climate)
print(fish_num_climate_d)

sample_num = len(fish_num_climate['fish_num'])
fish_num = fish_num_climate_d['fish_num']
weather_sunny = fish_num_climate_d['weather_sunny']
temperature = fish_num_climate_d['temperature']

stan_data = {
    'N': sample_num,
    'fish_num': fish_num,
    'sunny': weather_sunny,
    'temp': temperature
}

if os.path.exists('3-8-1-glm-pois-1.pkl'):
    sm = pickle.load(open('3-8-1-glm-pois-1.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-8-1-glm-pois-1.stan')

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

qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()

label_one = ['cloudy', 'sunny']
label_two = np.arange(1,31)

y_s = []
for i in label_two:
    y_s.append(np.exp(df['Intercept'].mean() + df['b_temp'].mean() * i + df['b_sunny'].mean() * 1))

y_c = []
for i in label_two:
    y_c.append(np.exp(df['Intercept'].mean() + df['b_temp'].mean() * i + df['b_sunny'].mean() * 0))

print(label_two.shape)
print(np.array(y_s).shape)
plt.plot(label_two, np.array(y_s), 'red', label='sunny')
plt.scatter(fish_num_climate_d.query('weather_sunny == 1')["temperature"],
            fish_num_climate_d.query('weather_sunny == 1')["fish_num"],
            c='r')

plt.plot(label_two, np.array(y_c), 'blue', label='cloudy')
plt.scatter(fish_num_climate_d.query('weather_cloudy == 1')["temperature"],
            fish_num_climate_d.query('weather_cloudy == 1')["fish_num"],
            c='b')

plt.legend()
plt.show()

print(fish_num_climate_d["fish_num"])

# saving compiled model
if not os.path.exists('3-8-1-glm-pois-1.pkl'):
    with open('3-8-1-glm-pois-1.pkl', 'wb') as f:
        pickle.dump(sm, f)