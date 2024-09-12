import pytest

import nsb.utils as utils


class TestSeedRandomStates:
    def test_seed_random_states(self):
        # Do a few checks, not just '42' because that's a common default
        for seed in range(5):
            utils.seed_random_states(seed)
            assert utils.np.random.get_state()[1][0] == seed
            assert utils.torch.initial_seed() == seed

    def test_random_reproducible(self):
        utils.seed_random_states(42)
        x = utils.np.random.rand()
        y = utils.torch.rand(1).item()

        utils.seed_random_states(42)
        assert x == utils.np.random.rand()
        assert y == utils.torch.rand(1).item()

    def test_random_not_reproducible(self):
        utils.seed_random_states(42)
        x = utils.np.random.rand()
        y = utils.torch.rand(1).item()

        utils.seed_random_states(43)
        assert x != utils.np.random.rand()
        assert y != utils.torch.rand(1).item()


class TestClamp:
    def test_clamp_below_min(self):
        assert utils.clamp(-1, 0, 5) == 0

    def test_clamp_above_max(self):
        assert utils.clamp(6, 0, 5) == 5

    def test_clamp_within_bounds(self):
        assert utils.clamp(3, 0, 5) == 3

    def test_clamp_prob_below_zero(self):
        assert utils.clamp_prob(-0.1) == 0.0

    def test_clamp_prob_above_one(self):
        assert utils.clamp_prob(1.1) == 1.0

    def test_clamp_prob_within_bounds(self):
        assert utils.clamp_prob(0.5) == 0.5


class TestDefaulter:
    def test_value_not_none(self):
        assert utils.defaulter(42, 0) == 42

    def test_value_is_none(self):
        assert utils.defaulter(None, 0) == 0


class TestFlattenValues:
    def test_flatten_values(self):
        timesteps = [0, 1, 2]
        y_values = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        repeated_timesteps, flattened_values = utils.flatten_values(timesteps, y_values)

        assert repeated_timesteps == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert flattened_values == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_flatten_values_longer_timesteps(self):
        timesteps = [0, 1, 2]
        y_values = [
            [1, 2, 3],
            [4, 5, 6]
        ]

        repeated_timesteps, flattened_values = utils.flatten_values(timesteps, y_values)
        assert repeated_timesteps == [0, 0, 0, 1, 1, 1]
        assert flattened_values == [1, 2, 3, 4, 5, 6]

    def test_flatten_values_longer_y_values(self):
        timesteps = [0, 1]
        y_values = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        repeated_timesteps, flattened_values = utils.flatten_values(timesteps, y_values)
        assert repeated_timesteps == [0, 0, 0, 1, 1, 1]
        assert flattened_values == [1, 2, 3, 4, 5, 6]



