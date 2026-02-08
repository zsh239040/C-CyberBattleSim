from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch as th
import gymnasium as gym

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import safe_mean, obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback


class BaselineMultiAgent:
    """On-policy SB3 agent wrapper for multi-agent rollouts."""

    def __init__(self, model: OnPolicyAlgorithm, logger: logging.Logger, role: str):
        self.model = model
        self.logger = logger
        self.role = str(role)
        self.callback: Optional[BaseCallback] = None

    @property
    def env(self) -> GymEnv:
        return self.model.env

    @property
    def num_timesteps(self) -> int:
        return self.model.num_timesteps

    @property
    def n_rollout_steps(self) -> int:
        return self.model.n_steps

    @property
    def log_interval(self) -> int:
        return 100 if self.model.__class__.__name__.lower() == "a2c" else 1

    @property
    def rollout_buffer(self) -> RolloutBuffer:
        return self.model.rollout_buffer

    def setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        eval_freq: int,
        n_eval_episodes: int,
        eval_log_path: Optional[str],
        reset_num_timesteps: bool,
        tb_log_name: str,
    ) -> Tuple[int, Optional[BaseCallback]]:
        callback: Optional[BaseCallback] = None
        if eval_env is not None and eval_freq > 0:
            from stable_baselines3.common.callbacks import EvalCallback

            callback = EvalCallback(
                eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                log_path=eval_log_path,
                warn=False,
            )

        total_timesteps, callback = self.model._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=False,
        )
        self.model.start_time = time.time()
        self.callback = callback
        if self.callback:
            self.callback.on_training_start(locals(), globals())
        return total_timesteps, callback

    def update_progress(self, total_timesteps: int):
        return self.model._update_current_progress_remaining(
            self.model.num_timesteps, total_timesteps
        )

    def on_rollout_start(self):
        assert self.model._last_obs is not None, "No previous observation was provided"
        self.model.policy.set_training_mode(False)
        self.rollout_buffer.reset()
        if self.model.use_sde:
            self.model.policy.reset_noise(self.env.num_envs)
        if self.callback:
            self.callback.on_rollout_start()

    def on_rollout_end(self, new_obs: Any, dones: Any):
        with th.no_grad():
            obs_tensor = obs_as_tensor(new_obs, self.model.device)
            try:
                _, values, _ = self.model.policy.forward(obs_tensor)
            except TypeError:
                _, values, _ = self.model.policy.forward(obs_tensor, action_masks=None)
        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if self.callback:
            self.callback.on_rollout_end()

    def on_training_end(self):
        if self.callback:
            self.callback.on_training_end()

    def perform_step(self, n_steps: int) -> Tuple[bool, Any, Any]:
        if self.model.use_sde and self.model.sde_sample_freq > 0 and n_steps % self.model.sde_sample_freq == 0:
            self.model.policy.reset_noise(self.env.num_envs)

        action_masks = None
        try:
            from sb3_contrib.common.maskable.utils import get_action_masks

            action_masks = get_action_masks(self.env)
        except Exception:
            action_masks = None

        with th.no_grad():
            obs_tensor = obs_as_tensor(self.model._last_obs, self.model.device)
            try:
                actions, values, log_probs = self.model.policy.forward(obs_tensor, action_masks=action_masks)
            except TypeError:
                actions, values, log_probs = self.model.policy.forward(obs_tensor)

        actions = actions.cpu().numpy()
        clipped_actions = actions
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        step_result = self.env.step(clipped_actions)
        if len(step_result) == 5:
            new_obs, rewards, terminated, truncated, infos = step_result
            dones = np.array([terminated or truncated]) if not isinstance(terminated, np.ndarray) else np.logical_or(terminated, truncated)
        else:
            new_obs, rewards, dones, infos = step_result

        self.model.num_timesteps += self.env.num_envs

        if self.callback:
            self.callback.update_locals(locals())
            if self.callback.on_step() is False:
                return False, new_obs, dones

        self.model._update_info_buffer(infos)

        if isinstance(self.model.action_space, gym.spaces.Discrete):
            actions = actions.reshape(-1, 1)

        self.rollout_buffer.add(
            self.model._last_obs,
            actions,
            rewards,
            self.model._last_episode_starts,
            values,
            log_probs,
        )
        self.model._last_obs = new_obs
        self.model._last_episode_starts = dones
        return True, new_obs, dones

    def train(self):
        self.model.train()

    def log_training(self, iteration: int):
        training_duration = time.time() - self.model.start_time
        fps = int(self.model.num_timesteps / training_duration) if training_duration > 0 else 0
        self.model.logger.record("time/iterations", iteration, exclude="tensorboard")

        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.model.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
            )
            self.model.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]),
            )

        self.model.logger.record("time/fps", fps)
        self.model.logger.record("time/time_elapsed", int(training_duration), exclude="tensorboard")
        self.model.logger.record("time/total_timesteps", self.model.num_timesteps, exclude="tensorboard")

        metrics = self._aggregate_episode_metrics()
        if metrics:
            self.model.logger.record(f"{self.role}/last_ep_reward", metrics["last_reward"])
            self.model.logger.record(f"{self.role}/last_ep_len", metrics["last_len"])
            self.model.logger.record(f"{self.role}/last10_avg_reward", metrics["avg_reward"])
            self.model.logger.record(f"{self.role}/last10_avg_len", metrics["avg_len"])

        self.model.logger.dump(step=self.model.num_timesteps)

    def save(self, filepath: str):
        self.model.save(filepath)
        self.logger.info("Saved model at: %s", filepath)

    def get_episode_metrics(self):
        return self._aggregate_episode_metrics()

    def _aggregate_episode_metrics(self):
        env = self.env
        wrappers = getattr(env, "envs", [env])

        last_rewards = []
        last_lens = []
        avg_rewards = []
        avg_lens = []
        last_action_counts = []
        avg_action_counts = []
        last_invalid_counts = []
        avg_invalid_counts = []
        last_action_values = []
        avg_action_values = []
        last_owned_ratios = []
        avg_owned_ratios = []

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

            last_action_count = getattr(w, "last_episode_action_count", None)
            if last_action_count is not None:
                last_action_counts.append(float(last_action_count))
            recent_action_counts = getattr(w, "recent_action_counts", None)
            if recent_action_counts:
                avg_action_counts.append(float(np.mean(recent_action_counts)))

            last_invalid = getattr(w, "last_invalid_action_count", None)
            if last_invalid is not None:
                last_invalid_counts.append(float(last_invalid))
            recent_invalid = getattr(w, "recent_invalid_action_counts", None)
            if recent_invalid:
                avg_invalid_counts.append(float(np.mean(recent_invalid)))

            last_action_value = getattr(w, "last_action_value", None)
            if last_action_value is not None:
                last_action_values.append(float(last_action_value))
            recent_action_values = getattr(w, "recent_action_values", None)
            if recent_action_values:
                avg_action_values.append(float(np.mean(recent_action_values)))

            last_owned_ratio = getattr(w, "last_owned_ratio", None)
            if last_owned_ratio is not None:
                last_owned_ratios.append(float(last_owned_ratio))
            recent_owned_ratios = getattr(w, "recent_owned_ratios", None)
            if recent_owned_ratios:
                avg_owned_ratios.append(float(np.mean(recent_owned_ratios)))

        if not (last_rewards or avg_rewards or last_lens or avg_lens or last_action_counts or avg_action_counts or last_invalid_counts or avg_invalid_counts or last_action_values or avg_action_values or last_owned_ratios or avg_owned_ratios):
            return None

        def _safe_mean(values):
            return float(np.mean(values)) if values else float("nan")

        return {
            "last_reward": _safe_mean(last_rewards),
            "last_len": _safe_mean(last_lens),
            "avg_reward": _safe_mean(avg_rewards),
            "avg_len": _safe_mean(avg_lens),
            "last_action_count": _safe_mean(last_action_counts),
            "avg_action_count": _safe_mean(avg_action_counts),
            "last_invalid_count": _safe_mean(last_invalid_counts),
            "avg_invalid_count": _safe_mean(avg_invalid_counts),
            "last_action_value": _safe_mean(last_action_values),
            "avg_action_value": _safe_mean(avg_action_values),
            "last_owned_ratio": _safe_mean(last_owned_ratios),
            "avg_owned_ratio": _safe_mean(avg_owned_ratios),
        }
