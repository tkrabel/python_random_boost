import numpy as np

from scipy.stats import ortho_group

def gen_friedman_data(n_samples, 
                      n_inputs, 
                      n_components=20, 
                      n_noise=0,
                      stn=1.0,
                      random_state=None):
    """Friedman (2001) random function generator
    
    Generates a data set from a random target function as described in
    Friedman (2001).
    
    Parameters
    ----------
    n_samples: int
        The number of observations in the data set.
    
    n_inputs: int
        The number of inputs considered to build the target function.

    n_components: int, optional (default=20)
        The number of components the formula has that computes the target.
        E.g. y = x1*x2 + x3^2 - x4 has thee components.

    n_noise: int, optional (default=0)
        The number of noise variables (follow a standard normal distribution).

    stn: float, optional (default=1.0)
        signal-to-noise-ratio that determines variance of error distribution. 

    random_state: int, optional (default=None)
        The random seed picked to produce random output. Use this for 
        reproducability

    Returns
    -------
    data: list of arrays, shape [(n_samples, n_inputs), (n_samples, )]
        Dataset containing input data and target function values of the input 
        samples.

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.
    """

    if random_state:
        np.random.seed(random_state)

    X = np.random.randn(n_samples, n_inputs)
    mus = X.mean(axis=0).reshape(1, n_inputs)
    nls = np.floor(np.random.exponential(scale=2, size=n_components) + 1.5)
    alphas = np.random.uniform(-1, 1, n_components)
    y = 0

    # For each component of function
    for l in range(n_components):
        # Number of features in component
        nl = int(nls[l]) if nls[l] <= n_inputs else n_inputs
        a = alphas[l]
        idx = np.random.choice(range(n_inputs), nl, False)
        Z = X[:, idx]
        mu = mus[:, idx]
        
        # Some Matrix magic (see Friedman (2001))
        U = ortho_group.rvs(dim=nl) if nl > 1 else 1
        D = np.diag(np.square(np.random.uniform(0.1, 2, nl)))
        V = U.dot(D).dot(U.T) if nl > 1 else D # V = U * D * U.T
        g = np.zeros(n_samples)

        for i in range(n_samples):
            g[i] = np.exp(-0.5 * np.diag((Z[i,:] - mu).dot(V).dot((Z[i,:] - mu).T)))
        y += a * g

    # Add error term according to signal-to-noise-ratio
    sd_norm = stn * np.mean(np.abs(y - np.median(y)))
    y += np.random.normal(0, sd_norm, size=n_samples)

    # Add noise variables to design matrix
    E = np.random.randn(n_samples, n_noise)
    X = np.hstack((X, E))

    return [X, y]
