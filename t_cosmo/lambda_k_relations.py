import numpy as np
from scipy.special import gamma

n = 0.8

a_k_i = np.asarray(
    [
        [-0.349395404871373,  -1.0629526931221451,  -1.8203302183430596],
        [5.674192626267164,   10.648790155995867,   14.693288291531053 ],
        [0.2957502016916985,  1.8002037299447995,   2.741120521996333  ],
    ]
).T

def get_lambda_0_k(k, lambda_0_0):
    if k == 0:
        return lambda_0_0
    a_i = a_k_i[k - 1]
    x = lambda_0_0**(-1./5.)
    lam_terms = np.array([x, x**2, x**3])
    G_n_k = gamma(k + 10./(3. - n))/gamma(10./(3. - n))
    return G_n_k * lambda_0_0 * (1 + np.dot(a_i, lam_terms))
   
def get_lambda_from_mass(source_frame_mass, lambda_0_0, M0=1.4):
    """Based on Yagi & Yunes (2018) expansion (Eq. 22)"""
    lambda_tot = 0.0
    for k in range(4): # k in [0, 1, 2, 3]
        lambda_tot += get_lambda_0_k(k=k, lambda_0_0=lambda_0_0) * (1 - source_frame_mass/M0)**(k) / np.math.factorial(k)
    return lambda_tot
