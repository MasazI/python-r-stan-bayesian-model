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

file_beer_sales_3 = pandas.read_csv('3-6-1-beer-sales-3.csv')
print(file_beer_sales_3.head())

# violin plot
sns.violinplot(x='weather', y='sales', data=file_beer_sales_3)
plt.show()

file_beer_sales_3_dm = pandas.get_dummies(file_beer_sales_3)
print(file_beer_sales_3_dm.head())

sample_num = len(file_beer_sales_3_dm['sales'])
sales = file_beer_sales_3_dm['sales']
weather_cloudy = file_beer_sales_3_dm['weather_cloudy']
weather_rainy = file_beer_sales_3_dm['weather_rainy']
weather_sunny = file_beer_sales_3_dm['weather_sunny']

print(sample_num)
print(sales)
print(len(weather_sunny))

weather_rainy_pred = [0, 1, 0]
weather_sunny_pred = [0, 0, 1]

stan_data = {
    'N': sample_num,
    'sales': sales,
    'weather_rainy': weather_rainy,
    'weather_sunny': weather_sunny,
    'N_pred': 3,
    'weather_rainy_pred': weather_rainy_pred,
    'weather_sunny_pred': weather_sunny_pred
}

if os.path.exists('3-6-1-cat-lm.pkl'):
    sm = pickle.load(open('3-6-1-cat-lm.pkl', 'rb'))
    # sm = pystan.StanModel(file='3-6-1-cat-lm.stan')
else:
    # a model using prior for mu and sigma.
    sm = pystan.StanModel(file='3-6-1-cat-lm.stan')

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

# extracting predicted sales
mcmc_sample = mcmc_result.extract()

# box plot
df = pandas.DataFrame(mcmc_sample['sales_pred'])
print(df)
df.columns = ['cloudy', 'rainy', 'sunny']
df.plot.box()
plt.show()

# saving compiled model
if not os.path.exists('3-6-1-cat-lm.pkl'):
    with open('3-6-1-cat-lm.pkl', 'wb') as f:
        pickle.dump(sm, f)