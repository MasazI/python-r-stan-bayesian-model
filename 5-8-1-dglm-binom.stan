data {
    int T;
    int len_obs;
    int y[len_obs];
    int obs_no[len_obs];
}

parameters {
    vector[T] mu;
    real<lower=0> s_w;
}

model {
    for (i in 2:T) {
        mu[i] ~ normal(mu[i-1], s_w);
    }
    for (i in 1:len_obs) {
        y[i] ~ bernoulli_logit(mu[obs_no[i]]);
    }
}

generated quantities {
    vector[T] probs;
    probs = inv_logit(mu);
}