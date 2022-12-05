import time

import numpy as np
import torch
from torch.autograd.functional import hessian, jacobian


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

class Friedman(torch.nn.Module):
    """Friedman regression as a perceptron.

    Parameters
    ----------
    m_beta : int
        Numbers of beta weights.
    """
    def __init__(self,x):

        super().__init__()

        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self,x):

        return self.a*torch.sin(torch.pi*x[:,0]*x[:,1]) + self.b*(x[:,2]-self.c)**2 + self.d*(x[:,3]) + self.e* x[:,4]


def normal_eq_lr(X, y):
    """Normal Equation (Theoretical)

    Parameters
    ----------
    X : 2d tensor
        Features.
    y : 1d tensor
        Target.

    Returns
    -------
    betas : 1d array
        Estimated beta parameters.

    Notes
    -----
    Closed Form Solution
    beta_hat = (X^T %*% X)^-1 %*% X %*% y
    """
    A = np.linalg.inv(np.matmul(np.transpose(X), X))
    B = np.matmul(np.transpose(X), y)
    return np.matmul(A, B)


def train_model(X, y, method="sgd", model=None, lr=0.01, n_epochs=1000,
                seed=None, tol=None, tol_abs=None):
    """Train various gradient descent algorithms.

    Parameters
    ----------
    X : 2d tensor
        Features.
    y : 1d tensor
        Target.
    method : {'sgd', 'lbfgs', 'newton'}
        Optimizer type.
    model : {LinearRegression, Friedman, None}
        Initialized model to train. None defaults to LinearRegression.
    lr : float, optional, default: 0.01
        Learning rate (step size).
    n_epochs : int, optional, default: 1000
        Number of training iterations.
    seed : int, optional, default: None
        For repoducible random beta weights.
    tol : float, optional, default: None
        Early stopping tol.
    tol_abs : float, optional, default: None
        Absolute loss required for early stopping.

    Returns
    -------
    betas : 1d array
        Estimated beta parameters.
    loss_hist : 1d array
        Loss per step.
    elapsed : float
        Total time to train and run n_epochs.
    """

    # Set reproducible seed
    if seed is not None:
        torch.manual_seed(seed)

    # Initalize model and loss function
    if model is None:
        model = LinearRegression(len(X[0]))
    elif model.__name__ == 'Friedman':
        model = model(X)
    else:
        model = model(len(X[0]))

    betas = model.linear.weight.T.detach()
    loss_func = torch.nn.MSELoss()
    loss_hist = np.zeros(n_epochs)

    # Select optimizer
    if method == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif method == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    elif method != "newton":
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
        elif method == "lbfgs":
            loss = optimizer.step(closure)
        elif method == 'newton':

            # Compte loss
            loss = compute_loss(betas)

            # Compute gradient and Hessian
            grad = jacobian(compute_loss, betas)

            # Check how many features
            if X.size()[1] == 1:
                # Have to make a 1x1 tensor to be in unity with other cases when m > 1
                hess = torch.zeros((1, 1))
                hess[0] = hessian(compute_loss, betas).inverse().squeeze()
            else:
                hess = hessian(compute_loss, betas).squeeze().inverse()

            # Step
            betas = betas - torch.matmul(hess, grad)

        loss_hist[i] = loss

        # Catch exploding gradients (n < m for Newton's)
        if not np.isfinite(loss_hist[i]):
            loss_hist[:] = np.nan
            betas[:] = np.nan
            elapsed = np.nan
            return betas, loss_hist, elapsed

        # Early stopping
        if tol is not None:
            prev_loss = np.nan if i == 0 else loss_hist[i-1]
            abs_pass = abs(prev_loss-loss_hist[i]) <= tol
            if (tol_abs is None and abs_pass or
                tol_abs is not None and loss_hist[i] < tol_abs and abs_pass):
                loss_hist[i+1:] = np.nan
                break

    # Time
    end = time.time()
    elapsed = end - start

    if method != "newton":
        betas = model.linear.weight.detach()[0]
    else:
        betas = betas[:, 0]

    return betas, loss_hist, elapsed
