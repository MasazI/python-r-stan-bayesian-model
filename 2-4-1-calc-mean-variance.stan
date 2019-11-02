data {
    int N;              // size of samples
    vector[N] sales;    // data
}

parameters {
    real mu;                // mean
    real<lower=0> sigma;    // standard deviation
}

// not vectorized
// model {
//     // gaussian distribution according to mean mu, standard deviation sigma
//     for (i in 1:N) {
//         sales[i] ~ normal(mu, sigma);
//     }
// }

// vectorized
model {
    // gaussian distribution according to mean mu, standard deviation sigma
    sales ~ normal(mu, sigma);
}
