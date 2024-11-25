import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import newton
from scipy.optimize import minimize
import joblib

class OptimizationProblem:
    def __init__(self):
        self.iteration = 0

    def objective_fun(self, lamda, args):
        arg = dict(args)
        z = arg['z']
        z_max = np.max(z)
        epsilon = arg['epsilon']
        objective = lamda * epsilon + lamda * np.log(np.mean(np.exp((z-z_max)/lamda))) + z_max
        return objective

    def der1_phi(self, lamda, args):
        arg = dict(args)
        z = arg['z']
        z_max = np.max(z)
        epsilon = arg['epsilon']
        derivative = epsilon + np.log(np.mean(np.exp((z-z_max)/lamda))) + z_max/lamda \
                            - 1/lamda * np.sum(z*np.exp((z-z_max)/lamda))/np.sum(np.exp((z-z_max)/lamda))
        return derivative

def estimate_V(z, epsilon, kl_paras):
    pol_dict = {'z':z, 'epsilon': epsilon+5.2}
    lamda0, tol, max_iter, learning_rate = kl_paras['lamda0'], kl_paras['tol'], kl_paras['max_iter'], kl_paras['learning_rate']
    '''Update lamda'''
    problem = OptimizationProblem()
    new_lamda = minimize(problem.objective_fun,
                            x0=lamda0,
                            bounds=((1e-2, None), ),
                            options={"disp": True, "gtol": tol, "maxifun": max_iter},
                            jac=problem.der1_phi,
                            method='L-BFGS-B',
                            args=pol_dict.items())
    lamda_hat = new_lamda.x[0]
    V_hat = new_lamda.fun
    return lamda_hat, V_hat
def kl_nn(s1, s2, k=1):
    """
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Thanks for \cite{Lorenzo Pacchiardi} for providing this function'
    """

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
    s2_neighbourhood = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2)

    s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
    s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
    rho = s1_distances[:, -1]
    nu = s2_distances[:, -1]
    if np.any(rho == 0):
        warnings.warn(
            f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
            f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
            f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
            RuntimeWarning,
        )
    D = np.sum(np.log(nu / rho))

    return (d / n) * D + np.log(m / (n - 1))
