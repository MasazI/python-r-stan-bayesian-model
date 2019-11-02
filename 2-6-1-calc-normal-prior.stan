data {
    int N;              // size of samples
    vector[N] sales;    // data
}

parameters {
    real mu;                // mean
    real<lower=0> sigma;    // standard deviation
}

// not vectorized
model {
    mu ~ normal(0, 1000000);
    sigma ~ normal(0, 1000000);

    // gaussian distribution according to mean mu, standard deviation sigma
    for (i in 1:N) {
        sales[i] ~ normal(mu, sigma);
    }
}
