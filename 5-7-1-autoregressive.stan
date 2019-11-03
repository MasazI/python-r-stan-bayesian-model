data {
    int T;
    vector[T] y;
}

parameters {
    real beta;
    real<lower=0> sigma;
    real Intercept;
}

model {
    for (i in 2:T) {
        y[i] ~ normal(Intercept + beta * y[i-1], sigma);
    }
}