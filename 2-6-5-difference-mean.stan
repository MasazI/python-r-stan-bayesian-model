data {
    int N;              // size of samples
    vector[N] sales_a;    // sales of a
    vector[N] sales_b;    // sales of b
}

parameters {
    real mu_a;                // mean of a
    real<lower=0> sigma_a;    // standard deviation of a
    real mu_b;                // mean of b
    real<lower=0> sigma_b;    // standard deviation of b
}

model {
    // gaussian distribution according to mean mu, standard deviation sigma
    sales_a ~ normal(mu_a, sigma_a);
    sales_b ~ normal(mu_b, sigma_b);
}

generated quantities {
    real diff;
    diff = mu_b - mu_a;
}
