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


def normal_eq_lr(X, y):
    """
    ### Normal Equation (Theoretical)
    Closed Form Solution
    beta_hat = (X^T %*% X)^-1 %*% X %*% y
    """
    A = np.linalg.inv(np.matmul(np.transpose(X), X))
    B = np.matmul(np.transpose(X), y)
    return np.matmul(A, B)


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
