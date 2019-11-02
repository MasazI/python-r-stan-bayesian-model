data {
    int N;
    vector[N] sales;
    vector[N] publicity;
    vector[N] temperature;
}

parameters {
    real Intercept;
    real b_pub;
    real b_temp;
    real b_pub_temp;
    real<lower=0> sigma;
}

model {
    vector[N] mu = Intercept + b_pub*publicity + b_temp*temperature + b_pub_temp*publicity .* temperature;
    sales ~ normal(mu, sigma);
}