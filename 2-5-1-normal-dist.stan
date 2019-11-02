data {
    int N;
    vector[N] animal_num;
}

parameters {
    real<lower=0> mu;
    real<lower=0> sigma;
}

// vectorized
model {
    animal_num ~ normal(mu, sigma);
}

generated quantities {
    vector[N] posterior_predictive;
    for (i in 1:N) {
        posterior_predictive[i] = normal_rng(mu, sigma);
    }
}
