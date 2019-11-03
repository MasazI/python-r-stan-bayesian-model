data {
    int N;
    int fish_num[N];
    vector[N] temp;
    vector[N] human_id;
}

parameters {
    real Intercept;
    real b_temp;
    real b_human;
    real b_temp_human;
}

transformed parameters {
    vector[N] lambda = exp(Intercept + b_temp*temp + b_human*human_id + b_temp_human*temp .* human_id);
}

model {
    fish_num ~ poisson(lambda);
}
