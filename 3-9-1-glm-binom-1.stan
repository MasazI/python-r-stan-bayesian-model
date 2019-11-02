data {
    int N;
    int germination[N];
    int binom_size[N];
    vector[N] solar;
    vector[N] nutrition;
}

parameters {
    real Intercept;
    real b_solar;
    real b_nutrition;
}

model {
    vector[N] prob = inv_logit(Intercept + b_solar * solar + b_nutrition * nutrition);
    germination ~ binomial(binom_size, prob);
}
