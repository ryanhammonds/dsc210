"""Test dataset generation/simulations."""
import numpy as np
import torch
from optimization.datasets import sim_linear_model
from optimization.optimization import train_model, LinearRegression, normal_eq_lr



def test_LinearRegression():
    """Test LinearRegression."""

    m = 10
    X, _, _ = sim_linear_model(m * 10, m, seed=m)

    model = LinearRegression(m)
    b = model.linear.weight.detach().numpy()
    assert len(b.flatten()) == m

    # Accuracy check
    y_hat = model(X).detach().numpy()
    y_true = np.dot(X.numpy(), b.T).reshape(-1, 1)

    mse = ((y_true.flatten() - y_hat.flatten()) ** 2).mean()
    assert mse < .01


def test_normal_eq_lr():
    """Test normal equation."""
    m = 10
    X, beta, y = sim_linear_model(m * 10, m, seed=m)
    beta_hat = normal_eq_lr(X, y)
    mse = ((beta.flatten() - beta_hat.flatten()) ** 2).mean()
    assert mse < .01


def test_train_model():
    """Test 1 beta and 10 beta cases."""

    m = 1
    n_epochs = 500

    X, beta, y = sim_linear_model(m * 10, m, seed=m)

    for method in ["sgd", "newton", "lbfgs"]:

        beta_hat, loss_hist, elapsed = train_model(X, y, method=method, n_epochs=n_epochs)

        # Check convergence
        assert all(beta_hat.numpy().round(2) == beta.numpy().round(2))
        assert len(loss_hist) == n_epochs
        assert loss_hist[0] > 0
        assert isinstance(elapsed, float)
