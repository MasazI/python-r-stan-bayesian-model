data {
    int N;
    int animal_num[N];
}

parameters {
    real<lower=0> lambda;
}

// vectorized
model {
    animal_num ~ poisson(lambda);
}

generated quantities {
    int posterior_predictive[N];
    for (i in 1:N) {
        posterior_predictive[i] = poisson_rng(lambda);
    }
}
