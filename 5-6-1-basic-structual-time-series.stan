data {
    int T;
    vector[T] y;
}

parameters {
    real Intercept;
    vector[T] mu;
    vector[T] gamma;
    real<lower=0> sd_w;
    real<lower=0> sd_s;
    real<lower=0> sd_v;
}

transformed parameters {
    vector[T] alpha;
    for (i in 1:T) {
        alpha[i]  = mu[i] + gamma[i];
    }
}

model {
    for (i in 3:T) {
        mu[i] ~ normal(2 * mu[i-1] - mu[i-2], sd_w);
    }
    for (i in 7:T) {
        gamma[i] ~ normal(-sum(gamma[(i-6):(i-1)]), sd_s);
    }
    for (i in 1:T) {
        y[i] ~ normal(alpha[i], sd_v);
    }
}