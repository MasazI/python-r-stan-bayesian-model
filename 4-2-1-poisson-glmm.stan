data {
    int N;
    int fish_num[N];
    vector[N] sunny;
    vector[N] temp;
    int N_human;
    int human_id[N];
}

parameters {
    real Intercept;
    real b_temp;
    real b_sunny;

    vector[N_human] r;
    real<lower=0> sigma_r;
}

transformed parameters {
    vector[N] lambda;
    for (i in 1:N) {
        lambda[i] = exp(Intercept + b_temp*temp[i] + b_sunny*sunny[i] + r[human_id[i]]);
    }
}

model {
    r ~ normal(0, sigma_r);
    fish_num ~ poisson(lambda);
}
