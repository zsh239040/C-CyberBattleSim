"""Shared-reset environment wrapper to avoid double resets across attacker/defender."""
from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym


class SharedResetEnv(gym.Env):
    """Ensure only one underlying reset occurs between steps.

    When multiple wrappers share the same env, VecEnv can trigger multiple
    resets in the same logical episode. This wrapper makes reset idempotent
    until a step happens.
    """

    def __init__(self, env: gym.Env, proportional_cutoff_coefficient: float | None = None):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._reset_done_since_step = False
        self._last_obs = None
        self._last_info = {}
        self._episode_done_pending = False
        self._episode_done_reason = None
        self._proportional_cutoff_coefficient = (
            float(proportional_cutoff_coefficient)
            if proportional_cutoff_coefficient is not None
            else None
        )
        self._apply_proportional_cutoff()

    def _apply_proportional_cutoff(self):
        if self._proportional_cutoff_coefficient is None:
            return
        coeff = self._proportional_cutoff_coefficient
        if coeff <= 0:
            coeff = 0
        if hasattr(self.env, "set_proportional_cutoff_coefficient"):
            self.env.set_proportional_cutoff_coefficient(coeff)

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        if self._reset_done_since_step and self._last_obs is not None:
            self._apply_proportional_cutoff()
            return self._last_obs, dict(self._last_info)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_obs = obs
        self._last_info = dict(info) if isinstance(info, dict) else {}
        self._reset_done_since_step = True
        self._apply_proportional_cutoff()
        return obs, dict(self._last_info)

    def step(self, action):
        self._reset_done_since_step = False
        return self.env.step(action)

    def signal_episode_done(self, reason: str | None = None):
        """Mark that one agent ended the episode; the other should skip its next step."""
        self._episode_done_pending = True
        self._episode_done_reason = reason

    def consume_episode_done(self) -> tuple[bool, str | None]:
        """Return and clear the pending episode-done flag and reason."""
        if self._episode_done_pending:
            self._episode_done_pending = False
            reason = self._episode_done_reason
            self._episode_done_reason = None
            return True, reason
        return False, None

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


def unwrap_current_env(env: Any):
    """Return the current CyberBattleEnv instance behind wrappers.

    Handles SharedResetEnv and RandomSwitchEnv-like wrappers.
    """
    # unwrap gym wrappers
    while hasattr(env, "env"):
        env = env.env
    # RandomSwitchEnv keeps current_env
    if hasattr(env, "current_env"):
        return env.current_env
    return env
