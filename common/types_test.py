"""Test for types."""

from absl.testing import parameterized
from swirl_c.common import types
import tensorflow as tf


class TypesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for types."""

  def test_primitive_to_conservative_name_arbitrary(self):
    """Tests to evaluate conversion of an arbitrarily named primitive var."""
    primitive_name = 'b#*rho_brho_'
    expected = 'rho_b#*rho_brho_'
    result = types.primitive_to_conservative_name(primitive_name)
    self.assertEqual(result, expected)

  def test_primitive_to_conservative_name_rho(self):
    """Tests to confirm that the density name is unchanged."""
    primitive_name = types.RHO
    expected = 'rho'
    result = types.primitive_to_conservative_name(primitive_name)
    self.assertEqual(result, expected)

  def test_primitive_to_conservative_empty_name(self):
    """Tests to check an error is thrown for empty primitive name."""
    primitive_name = ''
    msg = r'Primitive variable name must not be empty\.'
    with self.assertRaisesRegex(ValueError, msg):
      types.primitive_to_conservative_name(primitive_name)

  def test_conservative_to_primitive_name_arbitrary(self):
    """Tests to evaluate conversion of an arbitrarily named conservative var."""
    primitive_name = 'b#*rho_brho_'
    conservative_name = 'rho_b#*rho_brho_'
    expected = primitive_name
    result = types.conservative_to_primitive_name(conservative_name)
    self.assertEqual(result, expected)

  def test_conservative_to_primitive_name_rho(self):
    """Tests to confirm that the density name is unchanged."""
    conservative_name = types.RHO
    expected = 'rho'
    result = types.conservative_to_primitive_name(conservative_name)
    self.assertEqual(result, expected)

  _BAD_CONSERVATIVE_VAR_NAMES = ('rho_', 'rhoU', 'Urho_', 'Urho_U')
  _GOOD_CONSERVATIVE_VAR_NAMES = ('rho_u', 'rho_e', 'rho_b#*rho_brho_')

  @parameterized.parameters(*zip(_BAD_CONSERVATIVE_VAR_NAMES))
  def test_conservative_to_primitive_incorrect_name(self, conservative_name):
    """Tests that an error is thrown for incorrectly named conservative var."""
    msg = (
        rf'Invalid conserved variable name: {conservative_name}. Must start'
        r' with "rho_".'
    )
    with self.assertRaisesRegex(ValueError, msg):
      types.conservative_to_primitive_name(conservative_name)

  @parameterized.parameters(*zip(_BAD_CONSERVATIVE_VAR_NAMES))
  def test_is_conservative_name_incorrect_name(self, conservative_name):
    """Tests that is_conservative_name returns `False` for incorrect format."""
    self.assertFalse(types.is_conservative_name(conservative_name))

  @parameterized.parameters(*zip(_GOOD_CONSERVATIVE_VAR_NAMES))
  def test_is_conservative_name_correct_name(self, conservative_name):
    """Tests that is_conservative_name returns `True` for correct format."""
    self.assertTrue(types.is_conservative_name(conservative_name))


if __name__ == '__main__':
  tf.test.main()
