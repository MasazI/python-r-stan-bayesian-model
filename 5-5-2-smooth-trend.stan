data {
    int T;
    vector[T] sales;
}

parameters {
    vector[T] mu;
    real<lower=0> s_w;
    real<lower=0> s_v;
}

model {
    for (i in 3:T) {
        mu[i] ~ normal(2*mu[i-1]-mu[i-2], s_w);
    }
    for (i in 1:T) {
        sales[i] ~ normal(mu[i], s_v);
    }
}