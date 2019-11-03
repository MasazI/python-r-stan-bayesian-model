data {
    int T;
    vector[T] y;
}

parameters {
    vector[T] mu;
    real<lower=0> s_w;
    real<lower=0> s_v;
}

model {
    for (i in 2:T) {
        mu[i] ~ normal(mu[i-1], s_w);
    }
    for (i in 2:T) {
        y[i] ~ normal(mu[i], s_v);
    }
}
