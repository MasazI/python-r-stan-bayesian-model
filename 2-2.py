###############
#
# Translate R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############

import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm

p = False

fish = pandas.read_csv('2-2-1-fish.csv')
print(fish)

# histgram
fish['length'].plot(kind='hist', bins=20)

# kernel density estimation
# pandas
fish['length'].plot(kind='kde', secondary_y=True)
if p:
    plt.show()

# kernel density estimation
# scikit-learn
from sklearn.neighbors import KernelDensity
# list
data_list = [x for x in fish["length"]]
# np.array
data_np = np.array(data_list)
# x軸をGridするためのデータも生成
x_grid = np.linspace(0, max(data_list), num=100)
weights = np.ones_like(data_list)/float(len(data_list))

bw = 0.2
kde_mode_bw = KernelDensity(bandwidth=bw, kernel='gaussian').fit(data_np[:, None])
score_bw = kde_mode_bw.score_samples(x_grid[:, None])
plt.hist(data_list, alpha=0.3, bins=20, weights=weights)
plt.plot(x_grid, np.exp(score_bw), label="origin")
plt.legend()
if p:
    plt.show()

# mean
print(fish['length'].mean())

# median, quantile
suuretu = np.arange(1001)
print(suuretu)

print(np.median(suuretu))

print(np.quantile(suuretu, q=0.5))

print(np.quantile(suuretu, q=0.25))

print(np.quantile(suuretu, q=0.75))

print(np.quantile(suuretu, q=0.025))

print(np.quantile(suuretu, q=0.975))

#covariance
birds = pandas.read_csv('2-1-1-birds.csv')
print(birds.head())

birds_corr = birds.corr()
print(birds_corr)

seaborn.heatmap(birds_corr, vmax=1, vmin=-1, center=0)
if p:
    plt.show()

# auto correlation
# pandas
nile = pandas.read_csv('Nile.csv')
print(nile)
nile_series = pandas.Series(nile['Nile'])
plt.plot(nile_series)
if p:
    plt.show()

auto_seires = [nile_series.autocorr(lag=i) for i in range(6)]
print(auto_seires)

# auto correlation
# statsmodels
auto_series_sm = sm.tsa.stattools.acf(nile_series, fft=False)
print(auto_series_sm)
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(111)
sm.graphics.tsa.plot_acf(nile_series, lags=40, ax=ax1)
if p:
    plt.show()
