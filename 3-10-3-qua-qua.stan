data {
    int N;
    vector[N] sales;
    vector[N] product;
    vector[N] clerk;
}

parameters {
    real Intercept;
    real b_pro;
    real b_cl;
    real b_pro_cl;
    real<lower=0> sigma;
}

model {
    vector[N] mu = Intercept + b_pro*product + b_cl*clerk + b_pro_cl*product .* clerk;
    sales ~ normal(mu, sigma);
}