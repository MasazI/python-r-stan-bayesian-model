data {
    int N;
    vector[N] sales;
    vector[N] temperature;

    int N_pred;
    vector[N_pred] temperature_pred;
}

parameters {
    real Intercept;
    real beta;
    real<lower=0> sigma;
}

model {
    sales ~ normal(Intercept + beta*temperature, sigma);
}

generated quantities {
    vector[N_pred] mu_pred;
    vector[N_pred] sales_pred;

    for (i in 1:N_pred) {
        mu_pred[i] = Intercept + beta*temperature_pred[i];
        sales_pred[i] = normal_rng(mu_pred[i], sigma);
    }
}


