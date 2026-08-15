"""RNG-layout contract for the exact 4-level x 8-stream Frontier arm.

Run with the final patched minimax clone's ``src`` first on ``PYTHONPATH``.
This verifies the mechanical prerequisite for conditionally independent
rollout evidence: copies of one level share their reset key/state, but receive
distinct environment-step keys and non-degenerate batched policy samples.
It does not claim that realized outcomes are statistically independent.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from minimax.envs.batch_env import BatchEnv


class _KeyEchoEnv:
    """Minimal functional environment exposing the keys BatchEnv supplies."""

    def reset(self, rng):
        return rng, {"reset_key": rng}, jnp.asarray(0, dtype=jnp.int32)

    def step(self, rng, state, action, reset_state, extra):
        del action, reset_state
        return rng, state, jnp.asarray(0.0), jnp.asarray(False), extra


def _make_key_echo_batch(n_parallel=4, n_eval=8):
    batch = object.__new__(BatchEnv)
    batch.n_parallel = n_parallel
    batch.n_eval = n_eval
    batch.sub_batch_size = n_parallel * n_eval
    batch.env = _KeyEchoEnv()
    return batch


class GroupedRNGContractTest(unittest.TestCase):
    def test_level_copies_share_reset_but_not_step_keys(self):
        n_parallel, n_eval = 4, 8
        batch = _make_key_echo_batch(n_parallel, n_eval)

        reset_obs, reset_state, _extra = batch._reset(
            jax.random.PRNGKey(1701), n_parallel, n_eval)
        reset_keys = np.asarray(reset_obs).reshape(n_parallel, n_eval, 2)
        state_keys = np.asarray(reset_state["reset_key"]).reshape(
            n_parallel, n_eval, 2)

        # Eight copies are the same task; the four task keys remain distinct.
        np.testing.assert_array_equal(reset_keys, state_keys)
        for level_keys in reset_keys:
            np.testing.assert_array_equal(
                level_keys, np.repeat(level_keys[:1], n_eval, axis=0))
        self.assertEqual(len(np.unique(reset_keys[:, 0], axis=0)), n_parallel)

        flat = n_parallel * n_eval
        state = {"reset_key": jnp.zeros((flat, 2), dtype=jnp.uint32)}
        step_obs, *_ = batch._step(
            jax.random.PRNGKey(1702),
            state,
            jnp.zeros((flat,), dtype=jnp.int32),
            state,
            jnp.zeros((flat,), dtype=jnp.int32),
        )
        step_keys = np.asarray(step_obs)
        self.assertEqual(step_keys.shape, (flat, 2))
        self.assertEqual(len(np.unique(step_keys, axis=0)), flat)

    def test_equal_policy_logits_do_not_share_one_action_draw(self):
        flat = 4 * 8
        policy = tfp.distributions.Categorical(logits=jnp.zeros((flat, 4)))
        actions = np.asarray(policy.sample(seed=jax.random.PRNGKey(1703)))
        self.assertEqual(actions.shape, (flat,))
        # This fixed-key check fails if a single scalar draw is broadcast to
        # all 32 repeated streams under the pinned stack.
        self.assertGreater(len(np.unique(actions)), 1)


if __name__ == "__main__":
    unittest.main()
