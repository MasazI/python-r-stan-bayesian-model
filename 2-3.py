###############
#
# Translate R to Python Copyright (c) 2016 Masahiro Imai Released under the MIT license
#
###############
import pandas
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import seaborn

fish = pandas.read_csv('2-2-1-fish.csv')
print(fish.head())

kde = gaussian_kde(fish['length'])

fig = plt.figure(figsize=(10,5))

x_grid = np.linspace(0, max(fish['length']), num=100)
weights = np.ones_like(fish['length'])/float(len(fish['length']))

ax1 = fig.add_subplot(1, 2, 1)
ax1.hist(fish['length'], weights=weights)
ax1.set_xlabel('length')
ax1.set_ylabel('count')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.linspace(0, np.max(fish['length'])), kde(np.linspace(0, np.max(fish['length']))))
ax2.set_xlabel('length')
ax2.set_ylabel('density')

plt.show()

# histgram
fish['length'].plot(kind='hist', bins=20)

# kernel density estimation
# pandas
fish['length'].plot(kind='kde', secondary_y=True)

plt.show()

# box plot
iris = pandas.read_csv('iris.csv')
print(iris.head())
seaborn.boxplot(x='Species', y='Petal.Length', data=iris)

plt.show()

# violin plot
seaborn.violinplot(x='Species', y='Petal.Length', data=iris)

plt.show()

# pair plot
seaborn.pairplot(data=iris, hue='Species', vars=['Petal.Length', 'Petal.Width'])
plt.show()

# line plot
nile = pandas.read_csv('Nile.csv')
print(nile.head())

seaborn.lineplot(data=nile, x='time', y='Nile')
plt.show()





