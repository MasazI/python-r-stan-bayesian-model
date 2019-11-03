###############
#
# Transform R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
#
###############

import numpy as np
import matplotlib.pyplot as plt
import pandas


def plot_ssm(mcmc_sample, time_vec, title, y_label, state_name, obs_vec=None):
    '''
    plot ssm
    :param mcmc_sample: Dataframe of pandas
    :param time_vec: time vector as x-axis
    :param title: graph title
    :param y_label: y label
    :param state_name: variable name of state
    :param obs_vec: observation's value
    :return:
    '''
    print(mcmc_sample)
    print(mcmc_sample[state_name].T.shape)

    print(np.quantile(mcmc_sample[state_name].T, 0.025))

    mu_df = pandas.DataFrame(mcmc_sample[state_name])

    print(mu_df.head())

    mu_quantile = mu_df.quantile(q=[0.025, 0.5, 0.975])
    print(mu_quantile)

    mu_quantile.index = ['lwr', 'fit', 'upr']
    mu_quantile.columns = pandas.Index(time_vec)
    print(mu_quantile)

    y1 = mu_quantile.iloc[0].values
    y2 = mu_quantile.iloc[1].values
    y3 = mu_quantile.iloc[2].values

    plt.fill_between(time_vec, y1, y3, facecolor='blue', alpha=0.1)
    plt.plot(time_vec, y2, 'blue', label=state_name)
    if obs_vec is not None:
        plt.scatter(time_vec, obs_vec, s=5)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('time')
    plt.legend()
    plt.tight_layout()
    plt.show()