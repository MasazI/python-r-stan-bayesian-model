data {
    int N;
    vector[N] sales;
    int weather_rainy[N];
    int weather_sunny[N];
    vector[N] temperature;
    int N_pred_w;
    int weather_rainy_pred[N_pred_w];
    int weather_sunny_pred[N_pred_w];
    int N_pred_t;
    vector[N_pred_t] temperature_pred;
}

parameters {
    real Intercept;
    real beta_weather_rainy;
    real beta_weather_sunny;
    real beta_temperature;
    real<lower=0> sigma;
}

model {
    for (i in 1:N) {
        real mu = Intercept + beta_weather_rainy*weather_rainy[i] + beta_weather_sunny*weather_sunny[i] + beta_temperature*temperature[i];
        sales[i] ~ normal(mu, sigma);
    }
}

generated quantities {
    matrix[N_pred_w, N_pred_t] mu_pred;
    matrix[N_pred_w, N_pred_t] sales_pred;

    for (i in 1:N_pred_w) {
        for (j in 1:N_pred_t) {
            mu_pred[i, j] = Intercept + beta_weather_rainy*weather_rainy_pred[i] + beta_weather_sunny*weather_sunny_pred[i] + beta_temperature*temperature_pred[j];
            sales_pred[i, j] = normal_rng(mu_pred[i, j], sigma);
        }
    }
}