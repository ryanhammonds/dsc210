import time

import numpy as np
import torch
from numpy.linalg import norm
from torch.autograd.functional import hessian, jacobian
from tqdm.notebook import tqdm


def sim_lm(n_obs, m_betas, X=None, b=None, seed=None):
    """Simulate a linear model as Xb = y.

    Parameters
    ----------
    n_obs : int
        Number of observatins (rows of X).
    m_beta : int
        Numbers of beta weights.
    X : 2d array, optional, default: None
        Features. Defaults to random normal.
    b : 1d array, optional default: None
        Beta weights. Default to random normal.
    seed : int, optional, default: None
        Random seed for reproducibility.

    Returns
    -------
    X : 2d tensor
        Features.
    b : 1d tensor
        Beta weights.
    y : 1d tensor
        Target.
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Initalize arrays
    if X is None:
        X = np.random.normal(size=(n_obs, m_betas))

    if b is None:
        b = np.random.normal(size=m_betas)

    # Solve y that corresponds to b and X
    y = np.dot(X, b).reshape(-1, 1)

    # Torch's linear layers require f32 tensors
    X = X.astype(np.float32)
    b = b.astype(np.float32)
    y = y.astype(np.float32)

    # Create tensors
    X = torch.from_numpy(X)
    b = torch.from_numpy(b)
    y = torch.from_numpy(y)

    return X, b, y


def normal_eq_lr(X, y):
    """
    ### Normal Equation (Theoretical)
    Closed Form Solution
    beta_hat = (X^T %*% X)^-1 %*% X %*% y
    """
    A = np.linalg.inv(np.matmul(np.transpose(X), X))
    B = np.matmul(np.transpose(X), y)
    return np.matmul(A, B)


def newton_v1(X, y, max_iter=250):
    """Simulate a linear model as Xb = y.

    adapted from https://thatdatatho.com/newtons-method-bfgs-linear-regression/

    Parameters
    ----------
    X : 2d array, default: None
    Features.
    y : 1d array, default: None
    Observations.
    error: float, optional, default: 10**-5
    Stopping threshold for gradient norm. Defaults to 10**-5
    max_iter: int, optional, default: 250
    Maximum iterations for Newton's method. If solution has not been found, alg stops.

    Returns
    -------
    (betas,MSEs)
    betas : m x 1 tensor
        Solution for least squares.
    MSEs_to_npy : numpy array
        MSE of each y_hat from the beta solution for each iteration. If there is no convergence on the kth step
        The MSE of the kth step and any other step until the max_iterth step is nan.
    """
    # Convert from tensor to numpy array
    X = np.array(X)

    m = X.shape[1]
    n = y.shape[0]

    beta = np.zeros((m, 1))  # Start with a guess
    A = -np.transpose(X)
    B = y - np.matmul(X, beta)
    gradient = np.matmul(A, B)  # 2 * t(X) %*% X %*% beta
    hessian = np.matmul(np.transpose(X), X)  # 2 * t(X) %*% X

    # Start Newton's method implementation here:
    k = 0
    grads = dict()
    MSEs = dict()
    while k < max_iter:  # stop if exceeds max iterations allowed
        A = -np.transpose(X)
        B = y - np.matmul(X, beta)
        gradient = np.matmul(A, B)
        try:
            # If norm of gradient cannot be calculated because there's an infinite in there
            grads[k] = norm(gradient)
            # There is no convergence.
            hessian = np.matmul(np.transpose(X), X)
            search = np.matmul(np.linalg.inv(-hessian), gradient).numpy()
            beta = beta + search  # beta - inv(-hessian) %*% gradient
            y_hat = torch.tensor(
                np.matmul(X, beta)
            )  # calculate y_hat from beta solution for each iteration tf.convert_to_tensor(numpy_array)
            loss_func = torch.nn.MSELoss()
            MSEs[k] = loss_func(y_hat, y)
            k += 1
        except:
            while k < max_iter:
                MSEs[k] = np.nan  # Since it breaks on this kth step, make this MSE NA
                k += 1  # Also do this for any step after this step.
            MSEs_to_npy = np.array([z[1] for z in MSEs.items()])
            res = (
                torch.from_numpy(np.full((m, 1), np.nan)),
                MSEs_to_npy,
            )  # Return no solution and MSEs
            # Not sure if this function should just return the beta solution in a previous epoch that worked...
            return res
            break
    MSEs_to_npy = np.array([z[1] for z in MSEs.items()])
    res = (torch.from_numpy(beta), MSEs_to_npy)
    return res


class LinearRegression(torch.nn.Module):
    """Muliple linear regression as a perceptron.

    Parameters
    ----------
    m_beta : int
        Numbers of beta weights.
    """

    def __init__(self, m_betas):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(m_betas, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class LinearRegression(torch.nn.Module):
    """Muliple linear regression as a perceptron.

    Parameters
    ----------
    m_beta : int
        Numbers of beta weights.
    """

    def __init__(self, m_betas):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(m_betas, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def train_model(X, y, method="sgd", lr=0.01, n_epochs=1000, opt_kwargs=None):
    """Train various gradient descent algorithms.

    Parameters
    ----------
    X : 2d tensor
        Features.
    y : 1d tensor
        Target.
    method : {'sgd', 'lbfgs', 'newton'}
        Model to train.
    lr : float, optional, default: 0.01
        Learning rate (step size).
    n_epochs : int, optional, default: 1000
        Number of training iterations.
    opt_kwargs : dict, optional, default: None
        Optimizers kwargs to pass through to pytorch.

    Returns
    -------
    betas : 1d array
        Estimated beta parameters.
    loss_hist : 1d array
        Loss per step.
    elapsed : float
        Total time to train and run n_epochs.
    """

    # Initalize model and loss function
    model = LinearRegression(len(X[0]))
    loss_func = torch.nn.MSELoss()
    loss_hist = np.zeros(n_epochs)

    # Select optimizer
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs
    if method == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, **opt_kwargs)
    elif method == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, **opt_kwargs)
    elif method == "newton":
        betas = torch.rand(X.size(1), 1)
    else:
        raise ValueError("Undefined method.")

    # Required for LBFGS
    def closure():

        # Zero gradients
        optimizer.zero_grad()

        # Get predicted y
        y_hat = model(X)

        # Compute loss
        loss = loss_func(y_hat, y)

        # Update weights
        loss.backward()

        return loss

    # Required for Newton's method
    def compute_loss(betas):
        y_hat = torch.matmul(X, betas)
        return torch.nn.MSELoss()(y_hat, y)

    # Train
    start = time.time()
    for i in range(n_epochs):

        if method == "sgd":
            loss = closure()
            optimizer.step()
            loss_hist[i] = loss
        elif method == "lbfgs":
            loss = optimizer.step(closure)
            loss_hist[i] = loss
        elif method == "newton":

            # Compute gradient and Hessian
            grad = jacobian(compute_loss, betas)

            if X.size()[1] == 1:  # check how many features
                hess = torch.zeros(
                    (1, 1)
                )  # have to make a 1x1 tensor to be in unity with other cases when m > 1
                # for some reason, hessian() output has no shape so we need to put it in the initialized 1x1 tensor
                hess[0] = hessian(compute_loss, betas).inverse().squeeze()
            else:
                hess = hessian(compute_loss, betas).squeeze().inverse()

            # Step
            betas = betas - torch.matmul(hess, grad)

            loss_hist[i] = compute_loss(betas)

    # Time
    end = time.time()
    elapsed = end - start

    if method != "newton":
        betas = model.linear.weight.detach()[0]
    else:
        betas = betas[:, 0]

    return betas, loss_hist, elapsed


def test_train_model():
    """Test 1 beta and 10 beta cases."""

    m = 100

    times = {}

    X, beta, y = sim_lm(m * 10, m, seed=m)

    for method in ["sgd", "newton", "lbfgs"]:

        beta_hat, loss_hist, elapsed = train_model(X, y, method=method)

        times[method] = elapsed

        # Check convergence
        assert all(beta_hat.numpy().round(2) == beta.numpy().round(2))

    if m == 100:
        assert times["sgd"] < times["lbfgs"] < times["newton"]
