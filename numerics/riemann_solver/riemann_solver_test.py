"""Tests for riemann_solver.py."""

from swirl_c.core import parameter
from swirl_c.numerics.riemann_solver import hll
from swirl_c.numerics.riemann_solver import riemann_solver
from swirl_c.physics import physics_models as physics_models_lib
import tensorflow as tf


class RiemannSolverTest(tf.test.TestCase):

  def setUp(self):
    """Initializes the generic thermodynamics library object."""
    super().setUp()
    self.cfg = parameter.SwirlCParameters()
    self.physics_models = physics_models_lib.PhysicsModels(self.cfg)

  def test_riemann_connects_hll(self):
    """Function to check that the linker function correctly links to hll.py."""
    self.cfg.numeric_flux_scheme = 'HLL'
    result = riemann_solver.select_numeric_flux_fn(
        self.cfg,
    )
    self.assertIs(result, hll.hll_convective_flux)

  def test_riemann_raises_for_bad_opt(self):
    """Function to check ValueError is raised if scheme not implemented."""
    self.cfg.numeric_flux_scheme = 'bad_opt'
    msg = r'^("bad_opt" is not implemented)'
    with self.assertRaisesRegex(ValueError, msg):
      riemann_solver.select_numeric_flux_fn(self.cfg)


if __name__ == '__main__':
  tf.test.main()
