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

file_beer_sales_4 = pandas.read_csv('3-7-1-beer-sales-4.csv')
print(file_beer_sales_4.head())

sns.scatterplot(
    x='temperature',
    y='sales',
    hue='weather',
    data=file_beer_sales_4
)
plt.show()

file_beer_sales_4_d = pandas.get_dummies(file_beer_sales_4)

print(file_beer_sales_4_d.head())

sample_num = len(file_beer_sales_4_d['sales'])
sales = file_beer_sales_4_d['sales']
weather_rainy = file_beer_sales_4_d['weather_rainy']
weather_sunny = file_beer_sales_4_d['weather_sunny']
temperature = file_beer_sales_4_d['temperature']

# for pred
# cloudy, rainy, sunny
weather_rainy_pred = [0, 1, 0]
weather_sunny_pred = [0, 0, 1]
temperature_pred = range(11,31)

stan_data = {
    'N': sample_num,
    'sales': sales,
    'weather_rainy': weather_rainy,
    'weather_sunny': weather_sunny,
    'temperature': temperature,
    'N_pred_w': 3,
    'weather_rainy_pred': weather_rainy_pred,
    'weather_sunny_pred': weather_sunny_pred,
    'N_pred_t': len(temperature_pred),
    'temperature_pred': temperature_pred
}

if os.path.exists('3-7-1-cat-lm.pkl'):
    sm = pickle.load(open('3-7-1-cat-lm.pkl', 'rb'))
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-7-1-cat-lm.stan')

mcmc_result = sm.sampling(
    data=stan_data,
    seed=1,
    chains=4,
    iter=2000,
    warmup=1000,
    thin=1
)

print(mcmc_result)

mcmc_sample = mcmc_result.extract()
sales_pred = mcmc_sample['sales_pred']
print(type(sales_pred))
print(sales_pred.shape)

print(sales_pred.T.shape)
label_one = ['cloudy', 'rainy', 'sunny']
label_two = np.arange(11,31)
cols = pandas.MultiIndex.from_product([label_one, label_two])

df_c = pandas.DataFrame(sales_pred[:, 0, :])
df_r = pandas.DataFrame(sales_pred[:, 1, :])
df_s = pandas.DataFrame(sales_pred[:, 2, :])
df_c.columns = label_two
df_r.columns = label_two
df_s.columns = label_two

# visualization
qua = [0.025, 0.25, 0.50, 0.75, 0.975]
d_est = pandas.DataFrame()
pallet = ['green', 'blue', 'red']
dataframes = [df_c, df_r, df_s]
for j, label in enumerate(label_one):
    for i in label_two:
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
plt.legend()
plt.show()

# saving compiled model
if not os.path.exists('3-7-1-cat-lm.pkl'):
    with open('3-7-1-cat-lm.pkl', 'wb') as f:
        pickle.dump(sm, f)