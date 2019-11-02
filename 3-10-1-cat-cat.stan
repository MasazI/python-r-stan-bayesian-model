data {
    int N;
    vector[N] sales;
    vector[N] publicity;
    vector[N] bargen;
}

parameters {
    real Intercept;
    real b_pub;
    real b_bar;
    real b_pub_bar;
    real<lower=0> sigma;
}

model {
    vector[N] mu = Intercept + b_pub*publicity + b_bar*bargen + b_pub_bar*publicity .* bargen;
    sales ~ normal(mu, sigma);
}