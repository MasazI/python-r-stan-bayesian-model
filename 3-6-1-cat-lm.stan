data {
    int N;
    vector[N] sales;
    int weather_rainy[N];
    int weather_sunny[N];

    int N_pred;
    int weather_rainy_pred[N_pred];
    int weather_sunny_pred[N_pred];
}

parameters {
    real Intercept;
    real beta_weather_rainy;
    real beta_weather_sunny;
    real<lower=0> sigma;
}

model {
    for (i in 1:N) {
        real mu = Intercept + beta_weather_rainy*weather_rainy[i] + beta_weather_sunny*weather_sunny[i];
        sales[i] ~ normal(mu, sigma);
    }
}

generated quantities {
    vector[N_pred] mu_pred;
    vector[N_pred] sales_pred;

    for (i in 1:N_pred) {
        mu_pred[i] = Intercept + beta_weather_rainy*weather_rainy_pred[i] + beta_weather_sunny*weather_sunny_pred[i];
        sales_pred[i] = normal_rng(mu_pred[i], sigma);
    }
}