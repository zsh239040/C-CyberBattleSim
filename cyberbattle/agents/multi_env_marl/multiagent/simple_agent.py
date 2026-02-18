from __future__ import annotations

import os
import time
from typing import Any, Optional, Tuple

import numpy as np

from stable_baselines3.common.logger import Logger


class SimpleMultiAgent:
    """Lightweight agent for noop/random defender baselines."""

    def __init__(
        self,
        env,
        logger,
        role: str,
        mode: str,
        n_rollout_steps: int,
        sb3_logger: Optional[Logger] = None,
    ) -> None:
        self.env = env
        self.logger = logger
        self.role = str(role)
        self.mode = str(mode).lower()
        self._n_rollout_steps = int(n_rollout_steps)
        self.sb3_logger = sb3_logger
        self.num_timesteps = 0
        self._last_obs = None
        self._start_time = None

    @property
    def n_rollout_steps(self) -> int:
        return self._n_rollout_steps

    @property
    def log_interval(self) -> int:
        return 1

    def setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[Any],
        eval_freq: int,
        n_eval_episodes: int,
        eval_log_path: Optional[str],
        reset_num_timesteps: bool,
        tb_log_name: str,
    ) -> Tuple[int, None]:
        if self._last_obs is None:
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                self._last_obs, _info = reset_result
            else:
                self._last_obs = reset_result
        if reset_num_timesteps:
            self.num_timesteps = 0
        self._start_time = time.time()
        return int(total_timesteps), None

    def update_progress(self, total_timesteps: int):
        return None

    def on_rollout_start(self):
        return None

    def on_rollout_end(self, new_obs: Any, dones: Any):
        return None

    def on_training_end(self):
        return None

    def _sample_actions(self):
        num_envs = getattr(self.env, "num_envs", 1)
        if self.mode == "noop":
            return [[] for _ in range(num_envs)]
        actions = []
        for _ in range(num_envs):
            actions.append(self.env.action_space.sample())
        return actions

    def perform_step(self, n_steps: int) -> Tuple[bool, Any, Any]:
        if self._last_obs is None:
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                self._last_obs, _info = reset_result
            else:
                self._last_obs = reset_result

        actions = self._sample_actions()
        step_result = self.env.step(actions)
        if len(step_result) == 5:
            new_obs, _rewards, terminated, truncated, _infos = step_result
            if isinstance(terminated, np.ndarray):
                dones = np.logical_or(terminated, truncated)
            else:
                dones = np.array([terminated or truncated])
        else:
            new_obs, _rewards, dones, _infos = step_result

        num_envs = getattr(self.env, "num_envs", 1)
        self.num_timesteps += int(num_envs)
        self._last_obs = new_obs
        return True, new_obs, dones

    def train(self):
        return None

    def _aggregate_episode_metrics(self):
        wrappers = getattr(self.env, "envs", [self.env])
        last_rewards = []
        last_lens = []
        avg_rewards = []
        avg_lens = []

        for w in wrappers:
            last_reward = getattr(w, "last_episode_reward", None)
            last_len = getattr(w, "last_episode_len", None)
            if last_reward is not None:
                last_rewards.append(float(last_reward))
            if last_len is not None:
                last_lens.append(float(last_len))

            recent_rewards = getattr(w, "recent_episode_rewards", None)
            if recent_rewards:
                avg_rewards.append(float(np.mean(recent_rewards)))
            recent_lens = getattr(w, "recent_episode_lens", None)
            if recent_lens:
                avg_lens.append(float(np.mean(recent_lens)))

        if not (last_rewards or last_lens or avg_rewards or avg_lens):
            return None

        return {
            "last_reward": float(np.mean(last_rewards)) if last_rewards else float("nan"),
            "last_len": float(np.mean(last_lens)) if last_lens else float("nan"),
            "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else float("nan"),
            "avg_len": float(np.mean(avg_lens)) if avg_lens else float("nan"),
        }

    def log_training(self, iteration: int):
        if not self.sb3_logger:
            return
        training_duration = time.time() - (self._start_time or time.time())
        fps = int(self.num_timesteps / training_duration) if training_duration > 0 else 0
        self.sb3_logger.record("time/iterations", iteration, exclude="tensorboard")
        self.sb3_logger.record("time/fps", fps)
        self.sb3_logger.record("time/time_elapsed", int(training_duration), exclude="tensorboard")
        self.sb3_logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        metrics = self._aggregate_episode_metrics()
        if metrics:
            self.sb3_logger.record(f"{self.role}/last_ep_reward", metrics["last_reward"])
            self.sb3_logger.record(f"{self.role}/last_ep_len", metrics["last_len"])
            self.sb3_logger.record(f"{self.role}/last10_avg_reward", metrics["avg_reward"])
            self.sb3_logger.record(f"{self.role}/last10_avg_len", metrics["avg_len"])

        self.sb3_logger.dump(step=self.num_timesteps)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SimpleMultiAgent mode={self.mode}\n")

    def get_episode_metrics(self):
        return self._aggregate_episode_metrics()
