data {
    int N;
    vector[N] sales;
    vector[N] temperature;
}

parameters {
    real Intercept;
    real beta;
    real<lower=0> sigma;
}

model {
    sales ~ normal(Intercept + beta*temperature, sigma);
}
