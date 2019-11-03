###############
#
# Transform R to Python Copyright (c) 2019 Masahiro Imai Released under the MIT license
#
###############

import os
import matplotlib.pyplot as plt
import pystan
import pandas
import pickle
import seaborn as sns
import numpy as np
import tools

def mcmc(sales_df_2, filename):
    # publicity
    sns.lineplot(x="date", y="publicity", data=sales_df_2)
    plt.show()

    # sales
    sns.lineplot(x="date", y="sales", data=sales_df_2)
    plt.show()

    sales = sales_df_2['sales']
    T = len(sales)
    publicity = sales_df_2['publicity']

    stan_data = {
        'T': T,
        'sales': sales,
        'pub': publicity
    }

    if os.path.exists('%s.pkl' % filename):
        sm = pickle.load(open('%s.pkl' % filename, 'rb'))
        # sm = pystan.StanModel(file='5-4-1-simple-reg.stan')
    else:
        # a model using prior for mu and sigma.
        sm = pystan.StanModel(file='5-4-1-simple-reg.stan')

    control = {
        'adapt_delta': 0.8,
        'max_treedepth': 10
    }

    mcmc_result = sm.sampling(
        data=stan_data,
        seed=1,
        chains=4,
        iter=2000,
        warmup=1000,
        control=control,
        thin=1
    )

    print(mcmc_result)
    mcmc_result.plot()
    plt.show()

    # saving compiled model
    if not os.path.exists('%s.pkl' % filename):
        with open('%s.pkl' % filename, 'wb') as f:
            pickle.dump(sm, f)

    # regression line
    samples = mcmc_result.extract(permuted=True)
    intercept = np.mean(samples["Intercept"])
    beta = np.mean(samples["beta"])
    publicity_range = np.arange(0, 8)
    plt.plot(publicity_range, intercept + beta * publicity_range)
    plt.scatter(sales_df_2['publicity'], sales_df_2['sales'])
    plt.show()

# whole data
sales_df_2 = pandas.read_csv('5-4-1-sales-ts-2.csv')
print(sales_df_2.head())
print(sales_df_2.describe())

sales_df_2['date'] = pandas.to_datetime(sales_df_2['date'])
print(sales_df_2.head())
sales_df_2['date'] = sales_df_2['date'].view('int64') // 10 ** 9
print(sales_df_2.head())
mcmc(sales_df_2, "5-4-1-simple-reg")

# head of half
sales_df2_head = sales_df_2.head(50)
print(sales_df2_head.head())
mcmc(sales_df2_head, "5-4-1-simple-reg-head")

# tail of half
sales_df2_tail = sales_df_2.tail(50)
print(sales_df2_tail.tail())
mcmc(sales_df2_tail, "5-4-1-simple-reg-tail")
