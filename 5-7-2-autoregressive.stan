data {
    int T;
    vector[T] y;
}

parameters {
    real beta;
    real Intercept;
    vector[T] alpha;
    real<lower=0> sd_w;
    real<lower=0> sd_v;
}

model {
    for (i in 2:T) {
        alpha[i] ~ normal(Intercept + beta * alpha[i-1], sd_w);
    }

    for (i in 1:T) {
        y[i] ~ normal(alpha[i], sd_v);
    }
}