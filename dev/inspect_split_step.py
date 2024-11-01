import math
import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm, invgamma

def marginal_likelihood_normal_inverse_gamma(data, mu_0, lambda_, alpha, beta):
    n = len(data)
    x_bar = np.mean(data)
    s_squared = np.var(data, ddof=1)

    # Posterior parameters for the Normal-Inverse-Gamma
    alpha_post = alpha + n / 2
    beta_post = beta + 0.5 * (np.sum((data - x_bar) ** 2) + (lambda_ * n * (x_bar - mu_0) ** 2) / (lambda_ + n))
    # beta_post = beta + 0.5 * ((n - 1) * s_squared + (lambda_ * n * (x_bar - mu_0) ** 2) / (lambda_ + n))
    lambda_post = lambda_ + n

    # Compute the marginal likelihood
    term1 = gammaln(alpha_post) - gammaln(alpha)
    term2 = alpha * np.log(beta/beta_post)
    term3 = 0.5 * (np.log(lambda_) - np.log(lambda_post))
    term4 = -n / 2 * np.log(np.pi) - n * np.log(2)
    # print(term1,term2,term3,term4)

    marginal_log_likelihood = term1 + term2 + term3 + term4

    return marginal_log_likelihood


def main():

    mu_0 = 0
    pr = 15*15
    grid = np.arange(10)*500+2
    sigma2_app = 0.01
    # alpha = 10000000
    alpha = 10
    beta = sigma2_app*(alpha+1)
    lambda_ = 1.
    mu0 = 0.

    for nobs in grid:
        nreps = 100

        print("\n\n\n")
        count = 0
        for r in range(nreps):
            data = npr.randn(nobs).clip(-1,1)/10.
            data0,data1 = data[:nobs//2],data[nobs//2:]
            term0 = marginal_likelihood_normal_inverse_gamma(data0, mu_0, lambda_, alpha, beta)
            term1 = marginal_likelihood_normal_inverse_gamma(data1, mu_0, lambda_, alpha, beta)
            term2 = marginal_likelihood_normal_inverse_gamma(data, mu_0, lambda_, alpha, beta)
    
            gamma = math.lgamma(nobs/2) + math.lgamma(nobs/2) - math.lgamma(nobs)
            # hastings = gamma + term0 + term1 - term2
            hastings = term0 + term1 - term2
            # print(nobs, gamma, term0,term1,term2, hastings)
            count += hastings>0
        print("count: ",count)


        # -- [app] marginal likelihood --
        # term0 = math.lgamma(nobs/2) + math.lgamma(nobs/2) - math.lgamma(nobs)
        # term1 = math.lgamma(pr+nobs) - math.lgamma(pr) + math.log(1+nobs)/2. \
        #     - nobs/2.*math.log(math.pi) - nobs * math.log(2.)
        # _term2 = (np.sum(pix**2) - np.mean(pix)**2/(pr+nobs))/2.
        # term2 = pr * math.log(1.) - (pr+nobs)*math.log(_term2.item())
        # print(nobs, term0, term1, term2, term0 + term1 + term2)

        # -- [app] marginal likelihood --
        # term0 = math.lgamma(nobs/2) + math.lgamma(nobs/2) - math.lgamma(nobs)
        # term1 = math.lgamma(alpha+nobs/2.) - math.lgamma(alpha) \
        #     + math.log(1./(1+nobs))/2. \
        #     - nobs/2.*math.log(math.pi) - nobs * math.log(2.)
        # _term2 = beta+(mu0/kappa+np.sum(pix**2) - np.mean(pix)**2/(kappa+nobs))/2.
        # term2 = alpha * math.log(beta) - (alpha+nobs)*math.log(_term2.item())
        # print(nobs, term0, term1, term2, term0 + term1 + term2)


        # -- [shape] marginal likelihood --

        # print(nobs, term0, term1, term0 + term1)

if __name__ == "__main__":
    main()
