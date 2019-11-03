data {
    int N;
    int fish_num[N];
    vector[N] sunny;
    vector[N] temp;

    int N_pred_s;
    vector[N_pred_s] sunny_pred;

    int N_pred_t;
    vector[N_pred_t] temp_pred;
}

parameters {
    real Intercept;
    real b_temp;
    real b_sunny;
}

model {
    vector[N] lambda = exp(Intercept + b_temp*temp + b_sunny*sunny);
    fish_num ~ poisson(lambda);
}

generated quantities {
    matrix[N_pred_s, N_pred_t] lambda_pred;
    matrix[N_pred_s, N_pred_t] fish_num_pred;


    for (i in 1:N_pred_s) {
        for (j in 1:N_pred_t) {
            lambda_pred[i, j] = Intercept + b_sunny*sunny_pred[i]+ b_temp*temp_pred[j];
            fish_num_pred[i, j] = poisson_log_rng(lambda_pred[i, j]);
        }
    }
}