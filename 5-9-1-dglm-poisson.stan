data {
    int T;
    int y[T];
    vector[T] ex;
}

parameters {
    real<lower=0> s_r;
    real<lower=0> s_g;
    vector[T] mu;
    real beta;
    vector[T] r;
}

transformed parameters {
    vector[T] lambda;
    for (i in 1:T) {
        lambda[i] = mu[i] + beta * ex[i] + r[i];
    }
}

model {
    r ~ normal(0, s_r);
    for (i in 3:T) {
        mu[i] ~ normal(2*mu[i-1] - mu[i-2], s_g);
    }
    y ~ poisson_log(lambda);
}

generated quantities {
    vector[T] lambda_exp;
    vector[T] lambda_smooth;
    vector[T] lambda_smooth_fix;

    lambda_exp = exp(lambda); // expectation of state.
    lambda_smooth = exp(mu + beta*ex); // expectation of state without r.
    lambda_smooth_fix = exp(mu + beta * mean(ex)); // expectation of state and mean ex
}