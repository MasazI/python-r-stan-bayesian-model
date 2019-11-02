data {
    int N;
    int fish_num[N];
    vector[N] sunny;
    vector[N] temp;
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
