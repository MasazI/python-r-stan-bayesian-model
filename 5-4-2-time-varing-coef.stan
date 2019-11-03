data {
    int T;
    vector[T] ex;
    vector[T] y;
}

parameters {
    vector[T] mu;
    vector[T] b;
    real<lower=0> s_w;
    real<lower=0> s_t;
    real<lower=0> s_v;
}

transformed parameters {
    vector[T] alpha;
    for (i in 1:T) {
        alpha[i] = mu[i] + b[i] * ex[i];
    }
}

model {
    for (i in 2:T) {
        mu[i] ~ normal(mu[i-1], s_w);
        b[i] ~ normal(b[i-1], s_t);
    }
    for (i in 1:T) {
        y[i] ~ normal(alpha[i], s_v);
    }
}