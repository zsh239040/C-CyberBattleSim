from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .shared_env import unwrap_current_env
from cyberbattle.simulation import model

import gymnasium as gym

from .environment_event_source import EnvironmentEventSource, IEnvironmentObserver
from .reward_store import IRewardStore


class AttackerEnvWrapper(gym.Env, IRewardStore, IEnvironmentObserver):
    """Lightweight attacker wrapper to coordinate resets and store rewards."""

    _log = logging.getLogger("cyberbattle.attacker")
    _summary_log = logging.getLogger("marl.episode")

    def __init__(
        self,
        cyber_env: gym.Env,
        event_source: Optional[EnvironmentEventSource] = None,
        max_timesteps: Optional[int] = None,
        log_episode_end: bool = False,
        log_episode_summary: bool = True,
        episode_log_prefix: str = "",
    ):
        super().__init__()
        self.cyber_env = cyber_env
        self.action_space = cyber_env.action_space
        self.observation_space = cyber_env.observation_space
        self.max_timesteps = max_timesteps
        self.timesteps: Optional[int] = None
        self.rewards: List[float] = []
        self.action_count = 0
        self.invalid_action_count = 0
        self.last_action = None
        self.last_action_value: Optional[float] = None
        self._log_episode_end = bool(log_episode_end)
        self._log_episode_summary = bool(log_episode_summary)
        self._episode_log_prefix = str(episode_log_prefix)
        self._episode_end_logged = False
        self._episode_end_recorded = False
        self._episode_summary_logged = False
        self.last_episode_reward: Optional[float] = None
        self.last_episode_len: Optional[int] = None
        self.recent_episode_rewards: Deque[float] = deque(maxlen=10)
        self.recent_episode_lens: Deque[int] = deque(maxlen=10)
        self.last_episode_action_count: Optional[int] = None
        self.recent_action_counts: Deque[int] = deque(maxlen=10)
        self.last_invalid_action_count: Optional[int] = None
        self.recent_invalid_action_counts: Deque[int] = deque(maxlen=10)
        self.recent_action_values: Deque[float] = deque(maxlen=10)
        self.last_owned_ratio: Optional[float] = None
        self.recent_owned_ratios: Deque[float] = deque(maxlen=10)
        self._action_counts: Dict[str, int] = {}
        self._owned_nodes_seen = set()
        self._failure_counts: Dict[str, int] = {}
        self.last_episode_failure_counts: Optional[Dict[str, int]] = None
        self.reset_request = False
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome: Optional[str] = None
        self._last_info: Dict[str, Any] = {}
        self._last_obs: Any = None

        if event_source is None:
            event_source = EnvironmentEventSource()
        self.event_source = event_source
        event_source.add_observer(self)

    @property
    def episode_rewards(self) -> List[float]:
        return self.rewards

    def on_reset(self, last_reward: float):
        self.reset_request = True
        self.last_reward = float(last_reward) if last_reward is not None else 0.0

    def _infer_episode_reason(self, fallback_reason: Optional[str] = None) -> str:
        if fallback_reason:
            return fallback_reason
        if self.last_outcome:
            return str(self.last_outcome)
        base_env = None
        try:
            base_env = unwrap_current_env(self.cyber_env)
        except Exception:
            base_env = None
        info_reason = None
        if isinstance(self._last_info, dict) and "end_episode_reason" in self._last_info:
            try:
                info_reason = int(self._last_info["end_episode_reason"])
            except Exception:
                info_reason = None
        if info_reason == 1:
            return "attacker_win"
        if info_reason == 2:
            return "attacker_lost"
        if info_reason == 3:
            if base_env is not None:
                try:
                    if getattr(base_env, "proportional_cutoff_coefficient", 0):
                        if hasattr(base_env, "proportional_cutoff_reached") and base_env.proportional_cutoff_reached():
                            return "proportional_cutoff"
                except Exception:
                    pass
                try:
                    if getattr(base_env, "num_iterations", 0) >= getattr(base_env, "episode_iterations", 0):
                        return "max_steps"
                except Exception:
                    pass
            return "cutoff"
        if base_env is not None:
            try:
                if hasattr(base_env, "defender_goal_reached") and base_env.defender_goal_reached():
                    return "defender_win"
            except Exception:
                pass
        if self.last_truncated:
            return "truncated"
        if self.last_terminated:
            return "terminated"
        return "unknown"

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self.timesteps is None:
            self.reset()

        pending = False
        pending_reason = None
        if hasattr(self.cyber_env, "consume_episode_done"):
            result = self.cyber_env.consume_episode_done()
            if isinstance(result, tuple):
                pending, pending_reason = result
            else:
                pending = bool(result)

        if pending:
            observation = self._last_obs
            reward = 0.0
            terminated = False
            truncated = True
            info = {"sync_skip": True}
            done = True
            self.rewards.append(float(reward))
            self.timesteps = int(self.timesteps or 0) + 1
            self.action_count += 1
            self.last_reward = float(reward)
            self.last_terminated = False
            self.last_truncated = True
            self._last_info = dict(info)
            if done and not self._episode_end_recorded:
                total_reward = float(sum(self.rewards)) if self.rewards else 0.0
                self.last_episode_reward = total_reward
                self.last_episode_len = int(self.timesteps)
                self.recent_episode_rewards.append(total_reward)
                self.recent_episode_lens.append(int(self.timesteps))
                self.last_episode_action_count = int(self.action_count)
                self.recent_action_counts.append(int(self.action_count))
                self.last_invalid_action_count = int(self.invalid_action_count)
                self.recent_invalid_action_counts.append(int(self.invalid_action_count))
                self._episode_end_recorded = True
            if done and self._log_episode_summary and not self._episode_summary_logged:
                actions_sorted = dict(sorted(self._action_counts.items(), key=lambda kv: (-kv[1], kv[0])))
                reason = self._infer_episode_reason(pending_reason)
                summary_owned_ratio = None
                try:
                    base_env = unwrap_current_env(self.cyber_env)
                    total_nodes = getattr(base_env, "num_nodes", 0)
                    if total_nodes:
                        summary_owned_ratio = float(len(self._owned_nodes_seen)) / float(total_nodes)
                except Exception:
                    summary_owned_ratio = None
                self._summary_log.info(
                    "ATTACKER EPISODE SUMMARY reward=%.3f len=%s reason=%s owned_ratio=%.4f invalid_actions=%d actions=%s",
                    float(self.last_episode_reward or 0.0),
                    str(self.last_episode_len),
                    reason,
                    float(summary_owned_ratio) if summary_owned_ratio is not None else -1.0,
                    int(self.last_invalid_action_count or 0),
                    actions_sorted,
                )
                self._episode_summary_logged = True
            return observation, float(reward), bool(terminated), bool(truncated), info

        self.last_action = np.array(action, copy=True) if action is not None else None
        try:
            self.last_action_value = float(np.linalg.norm(self.last_action)) if self.last_action is not None else None
        except Exception:
            self.last_action_value = None

        step_result = self.cyber_env.step(action)
        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
        else:
            observation, reward, done, info = step_result
            terminated, truncated = bool(done), False

        reward = float(reward)
        self.rewards.append(reward)
        self.timesteps = int(self.timesteps or 0) + 1
        self.action_count += 1
        invalid_action = False
        if isinstance(info, dict) and (info.get("invalid_action") or info.get("cyber_step_executed") is False):
            invalid_action = True
        if isinstance(info, dict) and info.get("outcome_class") is not None:
            try:
                if isinstance(info["outcome_class"], model.InvalidAction):
                    invalid_action = True
            except Exception:
                pass
        if isinstance(info, dict) and isinstance(info.get("outcome"), str):
            if "invalid" in info["outcome"].lower():
                invalid_action = True
        if invalid_action:
            self.invalid_action_count += 1

        label = "INVALID" if invalid_action else "UNKNOWN"
        if not invalid_action and isinstance(info, dict):
            outcome = info.get("outcome")
            vtype = info.get("vulnerability_type")
            if outcome is not None and vtype is not None:
                label = f"{vtype}:{outcome}"
            elif outcome is not None:
                label = str(outcome)
        self._action_counts[label] = self._action_counts.get(label, 0) + 1

        # Track failure outcomes that indicate defender impact or action failure.
        actual_outcome = None
        try:
            base_env = unwrap_current_env(self.cyber_env)
            actual_outcome = getattr(base_env, "outcome_obtained", None)
        except Exception:
            actual_outcome = None
        if actual_outcome is None and isinstance(info, dict):
            actual_outcome = info.get("outcome_class")
        if actual_outcome is not None:
            failure_types = (
                model.FirewallBlock,
                model.NonListeningPort,
                model.UnsuccessfulAction,
                model.NoVulnerability,
                model.NoEnoughPrivilege,
                model.OutcomeNonPresent,
                model.RepeatedResult,
                model.NonRunningMachine,
                model.InvalidAction,
            )
            for failure_type in failure_types:
                if isinstance(actual_outcome, failure_type):
                    name = failure_type.__name__
                    self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
                    break

        try:
            base_env = unwrap_current_env(self.cyber_env)
            if hasattr(base_env, "owned_nodes"):
                for node in base_env.owned_nodes:
                    self._owned_nodes_seen.add(node)
        except Exception:
            pass

        if self.reset_request:
            truncated = True

        if self.max_timesteps is not None and self.timesteps >= self.max_timesteps:
            truncated = True

        done = bool(terminated or truncated)
        self.last_reward = reward
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self._last_info = dict(info) if isinstance(info, dict) else {}

        if done and hasattr(self.cyber_env, "signal_episode_done"):
            self.cyber_env.signal_episode_done(self._infer_episode_reason())

        if done and not self._episode_end_recorded:
            total_reward = float(sum(self.rewards)) if self.rewards else 0.0
            self.last_episode_reward = total_reward
            self.last_episode_len = int(self.timesteps)
            self.recent_episode_rewards.append(total_reward)
            self.recent_episode_lens.append(int(self.timesteps))
            self.last_episode_action_count = int(self.action_count)
            self.recent_action_counts.append(int(self.action_count))
            self.last_invalid_action_count = int(self.invalid_action_count)
            self.recent_invalid_action_counts.append(int(self.invalid_action_count))
            if self.last_action_value is not None:
                self.recent_action_values.append(float(self.last_action_value))
            owned_ratio = None
            try:
                base_env = unwrap_current_env(self.cyber_env)
                if hasattr(base_env, "owned_nodes") and getattr(base_env, "num_nodes", 0):
                    owned_ratio = float(len(base_env.owned_nodes)) / float(base_env.num_nodes)
            except Exception:
                owned_ratio = None
            if owned_ratio is not None:
                self.last_owned_ratio = owned_ratio
                self.recent_owned_ratios.append(owned_ratio)
            self.last_episode_failure_counts = dict(self._failure_counts)
            self._episode_end_recorded = True

        if done and self._log_episode_summary and not self._episode_summary_logged:
            actions_sorted = dict(sorted(self._action_counts.items(), key=lambda kv: (-kv[1], kv[0])))
            reason = self._infer_episode_reason()
            summary_owned_ratio = None
            try:
                base_env = unwrap_current_env(self.cyber_env)
                total_nodes = getattr(base_env, "num_nodes", 0)
                if total_nodes:
                    summary_owned_ratio = float(len(self._owned_nodes_seen)) / float(total_nodes)
            except Exception:
                summary_owned_ratio = None
            self._summary_log.info(
                "ATTACKER EPISODE SUMMARY reward=%.3f len=%s reason=%s owned_ratio=%.4f invalid_actions=%d failures=%s actions=%s",
                float(self.last_episode_reward or 0.0),
                str(self.last_episode_len),
                reason,
                float(summary_owned_ratio) if summary_owned_ratio is not None else -1.0,
                int(self.last_invalid_action_count or 0),
                self.last_episode_failure_counts or {},
                actions_sorted,
            )
            self._episode_summary_logged = True

        if self._log_episode_end and done and not self._episode_end_logged:
            if self.last_outcome:
                reason = self.last_outcome
            elif self.last_truncated:
                reason = "truncated"
            elif self.last_terminated:
                reason = "terminated"
            elif self.max_timesteps is not None and self.timesteps >= self.max_timesteps:
                reason = "timeout"
            else:
                reason = "unknown"
            prefix = f"{self._episode_log_prefix} " if self._episode_log_prefix else ""
            total_reward = float(sum(self.rewards)) if self.rewards else 0.0
            self._log.warning(
                "%sATTACKER EPISODE END steps=%d reason=%s total_reward=%.3f",
                prefix,
                int(self.timesteps),
                reason,
                total_reward,
            )
            self._episode_end_logged = True

        self._last_obs = observation
        return observation, reward, bool(terminated), bool(truncated), info

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict[str, Any]]:
        if not self.reset_request:
            last_reward = self.rewards[-1] if self.rewards else 0.0
            self.event_source.notify_reset(last_reward)

        if (self.timesteps is not None and self.timesteps > 0 and not self._episode_end_recorded):
            total_reward = float(sum(self.rewards)) if self.rewards else 0.0
            self.last_episode_reward = total_reward
            self.last_episode_len = int(self.timesteps)
            self.recent_episode_rewards.append(total_reward)
            self.recent_episode_lens.append(int(self.timesteps))
            self.last_episode_action_count = int(self.action_count)
            self.recent_action_counts.append(int(self.action_count))
            self.last_invalid_action_count = int(self.invalid_action_count)
            self.recent_invalid_action_counts.append(int(self.invalid_action_count))
            if self.last_action_value is not None:
                self.recent_action_values.append(float(self.last_action_value))
            owned_ratio = None
            try:
                base_env = unwrap_current_env(self.cyber_env)
                if hasattr(base_env, "owned_nodes") and getattr(base_env, "num_nodes", 0):
                    owned_ratio = float(len(base_env.owned_nodes)) / float(base_env.num_nodes)
            except Exception:
                owned_ratio = None
            if owned_ratio is not None:
                self.last_owned_ratio = owned_ratio
                self.recent_owned_ratios.append(owned_ratio)
            self._episode_end_recorded = True

        reset_result = self.cyber_env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, info = reset_result
        else:
            observation, info = reset_result, {}
        self._last_obs = observation

        self.reset_request = False
        self.timesteps = 0
        self.rewards = []
        self.action_count = 0
        self.invalid_action_count = 0
        self.last_action = None
        self.last_action_value = None
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome = None
        self._episode_end_logged = False
        self._episode_end_recorded = False
        self._episode_summary_logged = False
        self._action_counts = {}
        self._owned_nodes_seen = set()
        self._failure_counts = {}
        self.last_episode_failure_counts = None

        return observation, dict(info) if isinstance(info, dict) else {}

    def close(self):
        return self.cyber_env.close()

    def render(self, *args, **kwargs):
        return self.cyber_env.render(*args, **kwargs)
