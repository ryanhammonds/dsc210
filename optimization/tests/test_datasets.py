"""Test dataset generation/simulations."""
import numpy as np
from optimization.datasets import sim_linear_model

def test_sim_linear_model():

    n_obs = 100
    m_betas = 10
    seed = 0

    X, b, y = sim_linear_model(n_obs, m_betas, X=None, b=None, seed=seed)
    assert X.detach().numpy().shape == (n_obs, m_betas)
    assert b.detach().numpy().shape == (m_betas,)

    _X, _b, _y = sim_linear_model(n_obs, m_betas, X=X.detach().numpy(), b=b.detach().numpy(), seed=seed)
    assert (X.detach().numpy() == _X.detach().numpy()).all()
    assert (b.detach().numpy() == _b.detach().numpy()).all()
    np.testing.assert_almost_equal(y.detach().numpy(), _y.detach().numpy(), decimal=3)
