data {
    int T;
    vector[T] sales;
    vector[T] pub;
}

parameters {
    real Intercept;
    real beta;
    real<lower=0> sigma;
}

transformed parameters {
    vector[T] mu = Intercept + beta * pub;
}

model {
    sales ~ normal(mu, sigma);
}