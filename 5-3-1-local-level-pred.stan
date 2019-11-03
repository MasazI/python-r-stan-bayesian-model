data {
    int T;
    vector[T] y;
    int pred_term;
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

generated quantities {
    vector[T + pred_term] mu_pred;
    mu_pred[1:T] = mu;
    for (i in 1:pred_term) {
        mu_pred[T + i] = normal_rng(mu_pred[T + i - 1], s_w);
    }
}