data {
    int N;
    int fish_num[N];
    vector[N] temp;

    int N_human;
    int human_id[N];
}

parameters {
    real Intercept;
    real b_temp;

    vector[N_human] t;
    real<lower=0> sigma_t;


    vector[N_human] r;
    real<lower=0> sigma_r;
}

transformed parameters {
    vector[N] lambda;
    for (i in 1:N) {
        lambda[i] = exp(Intercept + (b_temp + t[human_id[i]])*temp[i] + r[human_id[i]]);
    }
}

model {
    t ~ normal(0, sigma_t);
    r ~ normal(0, sigma_r);
    fish_num ~ poisson(lambda);
}
