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

fish_num_climate_2 = pandas.read_csv('4-1-1-fish-num-2.csv')
print(fish_num_climate_2.head())
print(fish_num_climate_2.describe())

sns.scatterplot(
    x='temperature',
    y='fish_num',
    hue='weather',
    data=fish_num_climate_2
)
plt.show()

fish_num_climate_2_d = pandas.get_dummies(fish_num_climate_2, columns=["weather", "id"])
print(fish_num_climate_2_d.head())

fish_num = fish_num_climate_2_d['fish_num']
sample_num = len(fish_num)
sunny = fish_num_climate_2_d['weather_sunny']
temperature = fish_num_climate_2_d['temperature']

sunny_pred = [0, 1]
N_pred_s = len(sunny_pred)
temperature_pred = range(0,31)
N_pred_t = len(temperature_pred)

stan_data = {
    'N': sample_num,
    'fish_num': fish_num,
    'sunny': sunny,
    'temp': temperature,
    'N_pred_s': N_pred_s,
    'sunny_pred': sunny_pred,
    'N_pred_t': N_pred_t,
    'temp_pred': temperature_pred
}

if os.path.exists('4-1-2-poisson.pkl'):
    sm = pickle.load(open('4-1-2-poisson.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='4-1-2-poisson.stan')

# using seed=1856510770 to avoid more than 20log2 witch is restriction of poisson_log function.
mcmc_result = sm.sampling(
    data=stan_data,
    chains=4,
    seed=1856510770,
    iter=2000,
    warmup=1000,
    thin=1
)

print(mcmc_result)
mcmc_result.plot()
plt.show()

print(mcmc_result.get_seed())

# saving compiled model
if not os.path.exists('4-1-2-poisson.pkl'):
    with open('4-1-2-poisson.pkl', 'wb') as f:
        pickle.dump(sm, f)

mcmc_sample = mcmc_result.extract()
fish_num_pred = mcmc_sample['fish_num_pred']

df_c = pandas.DataFrame(fish_num_pred[:, 0, :])
df_s = pandas.DataFrame(fish_num_pred[:, 1, :])
df_c.columns = temperature_pred
df_s.columns = temperature_pred

# visualization
qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()
pallet = ['blue', 'red']
dataframes = [df_c, df_s]

label_one = ['cloudy', 'sunny']
for j, label in enumerate(sunny_pred):
    for i in temperature_pred:
        for qu in qua:
            d_est[qu] = dataframes[j].quantile(qu)
    x = d_est.index
    y1 = d_est[0.025].values
    y2 = d_est[0.25].values
    y3 = d_est[0.5].values
    y4 = d_est[0.75].values
    y5 = d_est[0.975].values

    plt.fill_between(x,y1,y5,facecolor=pallet[j],alpha=0.1)
    plt.fill_between(x,y2,y4,facecolor=pallet[j],alpha=0.3)
    plt.plot(x,y3,pallet[j],label=label_one[j])

    plt.scatter(fish_num_climate_2_d.query('weather_sunny == %d' % j)["temperature"],
                fish_num_climate_2_d.query('weather_sunny == %d' % j)["fish_num"],
                c=pallet[j], label=label_one[j])
plt.legend()
plt.show()
