# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    callbacks_multi_env.py
    Callbacks and wrappers for multi-environment training and validation.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import gymnasium
from stable_baselines3.common.callbacks import BaseCallback
from cyberbattle.simulation import model as m


class EpisodeStatsWrapper(gymnasium.Wrapper):
    """Attach episode statistics to info when an episode ends."""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if done:
            try:
                stats = self.env.get_statistics()
            except Exception:
                stats = None
            info = dict(info)
            if stats is not None:
                info["episode_stats"] = stats
            info["env_id"] = getattr(self.env, "current_env_index", None)
        return observation, reward, terminated, truncated, info


class MultiEnvTrainingCallback(BaseCallback):
    """Aggregate training statistics across vectorized environments."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.verbose = verbose

    def _on_training_start(self) -> None:
        self.num_envs = self.training_env.num_envs
        self._init_env_buffers()

    def _init_env_buffers(self) -> None:
        self.local_remote_actions_counts = np.zeros((self.num_envs, 2))
        self.local_remote_actions_success = np.zeros((self.num_envs, 2))
        self.vulnerability_counts = [dict() for _ in range(self.num_envs)]
        self.vulnerability_success = [dict() for _ in range(self.num_envs)]
        self.outcome_actions_counts = [dict() for _ in range(self.num_envs)]
        self.target_node_tag_counts = [dict() for _ in range(self.num_envs)]
        self.source_nodes = [set() for _ in range(self.num_envs)]
        self.target_nodes = [set() for _ in range(self.num_envs)]
        self.min_distance_actions = [[] for _ in range(self.num_envs)]
        self.invalid_actions = np.zeros(self.num_envs)

    def _reset_env_buffers(self, env_idx: int) -> None:
        self.local_remote_actions_counts[env_idx] = 0
        self.local_remote_actions_success[env_idx] = 0
        self.vulnerability_counts[env_idx] = {}
        self.vulnerability_success[env_idx] = {}
        self.outcome_actions_counts[env_idx] = {}
        self.target_node_tag_counts[env_idx] = {}
        self.source_nodes[env_idx] = set()
        self.target_nodes[env_idx] = set()
        self.min_distance_actions[env_idx] = []
        self.invalid_actions[env_idx] = 0

    def _accumulate_step_metrics(self, env_idx: int, info: Dict[str, Any]) -> None:
        if not info:
            return
        if "source_node" in info:
            self.source_nodes[env_idx].add(info["source_node"])
        if "target_node" in info:
            self.target_nodes[env_idx].add(info["target_node"])
        if "vulnerability" in info:
            self.vulnerability_counts[env_idx][info["vulnerability"]] = (
                self.vulnerability_counts[env_idx].get(info["vulnerability"], 0) + 1
            )
        if "target_node_tag" in info:
            self.target_node_tag_counts[env_idx][info["target_node_tag"]] = (
                self.target_node_tag_counts[env_idx].get(info["target_node_tag"], 0) + 1
            )
        if "outcome" in info:
            self.outcome_actions_counts[env_idx][info["outcome"]] = (
                self.outcome_actions_counts[env_idx].get(info["outcome"], 0) + 1
            )
        if info.get("min_distance_action") is not None:
            self.min_distance_actions[env_idx].append(info["min_distance_action"])

        outcome_class = info.get("outcome_class")
        if outcome_class is not None and issubclass(type(outcome_class), m.InvalidAction):
            self.invalid_actions[env_idx] += 1
        elif "vulnerability" in info:
            self.vulnerability_success[env_idx][info["vulnerability"]] = (
                self.vulnerability_success[env_idx].get(info["vulnerability"], 0) + 1
            )

        vulnerability_type = info.get("vulnerability_type")
        if vulnerability_type == "local":
            self.local_remote_actions_counts[env_idx][0] += 1
            if outcome_class is not None and not issubclass(type(outcome_class), m.InvalidAction):
                self.local_remote_actions_success[env_idx][0] += 1
        elif vulnerability_type == "remote":
            self.local_remote_actions_counts[env_idx][1] += 1
            if outcome_class is not None and not issubclass(type(outcome_class), m.InvalidAction):
                self.local_remote_actions_success[env_idx][1] += 1

    def _get_episode_stats(self, env_idx: int, info: Dict[str, Any]) -> Optional[tuple]:
        stats = info.get("episode_stats")
        if stats is not None:
            return stats
        try:
            return self.training_env.env_method("get_statistics", indices=[env_idx])[0]
        except Exception:
            return None

    def _build_episode_metrics(self, env_idx: int, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        stats = self._get_episode_stats(env_idx, info)
        if stats is None:
            return None
        owned_nodes, discovered_nodes, not_discovered_nodes, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, reimaged_nodes, num_events, discovered_amount, discoverable_amount, _ = stats

        min_distance_actions = [x for x in self.min_distance_actions[env_idx] if x is not None]
        min_distance_mean = float(np.mean(min_distance_actions)) if min_distance_actions else 0.0

        end_reason = info.get("end_episode_reason")
        episode_goal_reached = 1 if end_reason == 1 else 0
        episode_lost = 1 if end_reason == 2 else 0

        owned_percentage = owned_nodes / num_nodes if num_nodes else 0
        discovered_percentage = discovered_nodes / num_nodes if num_nodes else 0
        reimaged_percentage = reimaged_nodes / num_nodes if num_nodes else 0
        disrupted_percentage = disrupted_nodes / num_nodes if num_nodes else 0

        reachable_count = reachable_count + 1
        discoverable_count = discoverable_count + 1
        disruptable_count = disruptable_count + 1

        discovered_amount_safe = discoverable_amount if discoverable_amount else 1
        discovered_nodes_safe = discovered_nodes if discovered_nodes else 1

        metrics = {
            "train/Owned nodes": owned_nodes,
            "train/Discovered nodes": discovered_nodes,
            "train/Discoverable amount": discoverable_amount,
            "train/Discovered amount": discovered_amount,
            "train/Not discovered nodes": not_discovered_nodes,
            "train/Total nodes": num_nodes,
            "train/Disrupted nodes": disrupted_nodes,
            "train/Reimaged nodes": reimaged_nodes,
            "train/Network availability": network_availability,
            "train/Number of events": num_events,
            "train/Minimum distance action": min_distance_mean,
            "train/Episode goal reached": episode_goal_reached,
            "train/Episode lost": episode_lost,
            "train/Owned nodes percentage": owned_percentage,
            "train/Discovered nodes percentage": discovered_percentage,
            "train/Discovered amount percentage": discovered_amount / discovered_amount_safe,
            "train/Reimaged nodes percentage": reimaged_percentage,
            "train/Disrupted nodes percentage": disrupted_percentage,
            "train/Relative owned nodes percentage": owned_nodes / reachable_count,
            "train/Relative discovered nodes percentage": discovered_nodes / discoverable_count,
            "train/Relative discovered amount percentage": (discovered_amount + 1) / discovered_amount_safe,
            "train/Relative disrupted nodes percentage": disrupted_nodes / disruptable_count,
            "train/Owned-discovered ratio": owned_nodes / discovered_nodes_safe,
            "train/Number of source nodes": len(self.source_nodes[env_idx]),
            "train/Number of target nodes": len(self.target_nodes[env_idx]),
            "train/Number of unique vulnerabilities": len(self.vulnerability_counts[env_idx]),
            "actions/train/Invalid actions": self.invalid_actions[env_idx],
            "actions/train/Success rate for all actions": sum(self.vulnerability_success[env_idx].values()),
            "actions/train/Total actions": sum(self.vulnerability_counts[env_idx].values()),
            "actions/train/Local actions count": self.local_remote_actions_counts[env_idx][0],
            "actions/train/Remote actions count": self.local_remote_actions_counts[env_idx][1],
            "actions/train/Success rate for local actions": self.local_remote_actions_success[env_idx][0],
            "actions/train/Success rate for remote actions": self.local_remote_actions_success[env_idx][1],
            "outcome_counts": self.outcome_actions_counts[env_idx],
            "target_node_tag_counts": self.target_node_tag_counts[env_idx],
        }
        return metrics

    def _log_aggregated_metrics(self, episodes_metrics: List[Dict[str, Any]]) -> None:
        num_episodes = len(episodes_metrics)
        aggregate: Dict[str, List[float]] = {}
        outcome_counts: Dict[str, float] = {}
        tag_counts: Dict[str, float] = {}
        for metrics in episodes_metrics:
            for key, value in metrics.items():
                if key == "outcome_counts":
                    for outcome, count in value.items():
                        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + count
                elif key == "target_node_tag_counts":
                    for tag, count in value.items():
                        tag_counts[tag] = tag_counts.get(tag, 0) + count
                else:
                    aggregate.setdefault(key, []).append(value)

        for key, values in aggregate.items():
            self.logger.record(key, float(np.mean(values)))
        for outcome, count in outcome_counts.items():
            self.logger.record(f"actions/train/Outcome {outcome} count", count / num_episodes)
        for tag, count in tag_counts.items():
            self.logger.record(f"actions/train/Target nodes {tag} count", count / num_episodes)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        finished_episodes: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            info = infos[env_idx] if env_idx < len(infos) else {}
            self._accumulate_step_metrics(env_idx, info)
            if env_idx < len(dones) and dones[env_idx]:
                metrics = self._build_episode_metrics(env_idx, info)
                if metrics is not None:
                    finished_episodes.append(metrics)
                self._reset_env_buffers(env_idx)

        if finished_episodes:
            self._log_aggregated_metrics(finished_episodes)
        return True


class MultiEnvValidationCallback(BaseCallback):
    """Validation callback with proper handling of vectorized envs (expects n_envs=1)."""

    def __init__(self, val_env, n_val_episodes, val_freq, validation_folder_path, output_logger, early_stopping=False, patience=10, verbose=0):
        super().__init__(verbose)
        self.val_env = val_env
        self.output_logger = output_logger
        self.n_val_episodes = n_val_episodes
        self.val_freq = val_freq
        self.validation_folder_path = validation_folder_path
        self.early_stopping = bool(early_stopping)
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.val_timesteps = 0
        self.current_patience = 0
        self.verbose = verbose

    def _on_step(self) -> bool:
        if (self.val_timesteps % self.val_freq) == 0:
            if self.verbose:
                self.output_logger.info(
                    "Validation phase at training step %s (set to be every %s)",
                    self.num_timesteps, self.val_freq
                )
            custom_metrics = self._run_evaluation()
            self._log_tensorboard_custom_metrics(custom_metrics)

            val_reward = custom_metrics["episode_reward"]
            if val_reward > self.best_mean_reward:
                self.model.save(f"{self.validation_folder_path}/checkpoint_{val_reward}_reward.zip")
                if self.verbose:
                    self.output_logger.info(
                        "Saving new best model with reward %s (previous best reward %s)",
                        val_reward, self.best_mean_reward
                    )
                self.best_mean_reward = val_reward
                self.current_patience = 0
            else:
                if self.verbose >= 2:
                    self.output_logger.info(
                        "Increasing patience: validation reward did not improve with %s (best reward %s)",
                        val_reward, self.best_mean_reward
                    )
                self.current_patience += 1
            if self.early_stopping and self.current_patience >= self.patience:
                if self.verbose:
                    self.output_logger.info(
                        "Stopping training due to lack of improvement in validation reward after %s patience checks",
                        self.patience
                    )
                return False

        self.val_timesteps += 1
        return True

    def _run_evaluation(self) -> Dict[str, float]:
        if getattr(self.val_env, "num_envs", 1) != 1:
            raise ValueError("MultiEnvValidationCallback expects a single validation environment.")

        local_actions_count_list = []
        local_actions_success_list = []
        remote_actions_count_list = []
        remote_actions_success_list = []
        source_nodes_list = []
        target_nodes_list = []
        vulnerability_counts_list = []
        vulnerability_success_list = []
        invalid_actions_list = []
        reachable_list = []
        discoverable_list = []
        discoverable_amount_list = []
        discovered_amount_list = []
        disruptable_list = []
        owned_list = []
        discovered_list = []
        disrupted_list = []
        reimaged_list = []
        not_discovered_list = []
        episode_reward_list = []
        owned_percentage_list = []
        discovered_percentage_list = []
        disrupted_percentage_list = []
        number_steps_list = []
        target_node_tag_counts_list = []
        outcome_actions_counts_list = []
        network_availability_list = []
        episode_won_list = []
        episode_lost_list = []
        reimaged_percentage_list = []
        num_events_list = []
        min_distance_actions_list = []

        for _ in range(self.n_val_episodes):
            local_remote_actions_counts = np.zeros(2)
            local_remote_actions_success = np.zeros(2)
            target_node_tag_counts = {}
            source_nodes = []
            target_nodes = []
            vulnerability_counts = {}
            vulnerability_success = {}
            outcome_actions_counts = {}
            invalid_actions = 0
            min_distance_actions = []

            obs = self.val_env.reset()
            episode_rewards = 0.0
            done = np.array([False])

            number_steps = 0
            info = None
            while not done[0]:
                number_steps += 1
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.val_env.step(action)
                episode_rewards += float(reward[0])
                info = info[0]
                if info.get("min_distance_action") is not None:
                    min_distance_actions.append(info["min_distance_action"])
                source_nodes.append(info["source_node"])
                source_nodes = list(set(source_nodes))
                target_nodes.append(info["target_node"])
                target_nodes = list(set(target_nodes))
                vulnerability_counts[info["vulnerability"]] = vulnerability_counts.get(info["vulnerability"], 0) + 1
                target_node_tag_counts[info["target_node_tag"]] = target_node_tag_counts.get(info["target_node_tag"], 0) + 1
                outcome_actions_counts[info["outcome"]] = outcome_actions_counts.get(info["outcome"], 0) + 1
                if issubclass(type(info["outcome_class"]), m.InvalidAction):
                    invalid_actions += 1
                else:
                    vulnerability_success[info["vulnerability"]] = vulnerability_success.get(info["vulnerability"], 0) + 1
                if info["vulnerability_type"] == "local":
                    local_remote_actions_counts[0] += 1
                    if not issubclass(type(info["outcome_class"]), m.InvalidAction):
                        local_remote_actions_success[0] += 1
                elif info["vulnerability_type"] == "remote":
                    local_remote_actions_counts[1] += 1
                    if not issubclass(type(info["outcome_class"]), m.InvalidAction):
                        local_remote_actions_success[1] += 1
            episode_won_list.append(info["end_episode_reason"] == 1)
            episode_lost_list.append(info["end_episode_reason"] == 2)
            local_actions_success_list.append(local_remote_actions_success[0])
            local_actions_count_list.append(local_remote_actions_counts[0])
            remote_actions_success_list.append(local_remote_actions_success[1])
            remote_actions_count_list.append(local_remote_actions_counts[1])
            source_nodes_list.append(len(set(source_nodes)))
            target_nodes_list.append(len(set(target_nodes)))
            vulnerability_counts_list.append(sum(vulnerability_counts.values()))
            vulnerability_success_list.append(sum(vulnerability_success.values()))
            invalid_actions_list.append(invalid_actions)
            outcome_actions_counts_list.append(outcome_actions_counts)
            target_node_tag_counts_list.append(target_node_tag_counts)
            min_distance_actions = [x for x in min_distance_actions if x is not None]
            if len(min_distance_actions) == 0:
                min_distance_actions_list.append(0)
            else:
                min_distance_actions_list.append(np.mean(min_distance_actions))

            stats = info.get("episode_stats")
            if stats is None:
                stats = self.val_env.envs[0].get_statistics()
            owned_nodes, discovered_nodes, not_discovered_nodes, disrupted_nodes, num_nodes, reachable_count, discoverable_count, disruptable_count, network_availability, reimaged_nodes, num_events, discovered_amount, discoverable_amount, _ = stats

            network_availability_list.append(network_availability)
            reachable_list.append(reachable_count + 1)
            discoverable_list.append(discoverable_count + 1)
            discovered_amount_list.append(discovered_amount + 1)
            discoverable_amount_list.append(discoverable_amount)
            disruptable_list.append(disruptable_count + 1)
            owned_percentage = owned_nodes / num_nodes if num_nodes else 0
            discovered_percentage = discovered_nodes / num_nodes if num_nodes else 0
            reimaged_percentage = reimaged_nodes / num_nodes if num_nodes else 0
            disrupted_percentage = disrupted_nodes / num_nodes if num_nodes else 0

            owned_list.append(owned_nodes)
            discovered_list.append(discovered_nodes)
            reimaged_list.append(reimaged_nodes)
            disrupted_list.append(disrupted_nodes)
            not_discovered_list.append(not_discovered_nodes)
            episode_reward_list.append(episode_rewards)
            owned_percentage_list.append(owned_percentage)
            discovered_percentage_list.append(discovered_percentage)
            disrupted_percentage_list.append(disrupted_percentage)
            number_steps_list.append(number_steps)
            reimaged_percentage_list.append(reimaged_percentage)
            num_events_list.append(num_events)

        stats = {
            "local_actions_count": np.mean(local_actions_count_list),
            "local_actions_success": np.mean(local_actions_success_list),
            "remote_actions_count": np.mean(remote_actions_count_list),
            "remote_actions_success": np.mean(remote_actions_success_list),
            "source_nodes": np.mean(source_nodes_list),
            "target_nodes": np.mean(target_nodes_list),
            "actions_counts": np.mean(vulnerability_counts_list),
            "actions_success": np.mean(vulnerability_success_list),
            "invalid_actions": np.mean(invalid_actions_list),
            "owned": np.mean(owned_list),
            "discovered": np.mean(discovered_list),
            "disrupted": np.mean(disrupted_list),
            "reachable": np.mean(reachable_list),
            "discoverable": np.mean(discoverable_list),
            "discoverable_amount": np.mean(discoverable_amount_list),
            "discovered_amount": np.mean(discovered_amount_list),
            "disruptable": np.mean(disruptable_list),
            "not_discovered": np.mean(not_discovered_list),
            "episode_reward": np.mean(episode_reward_list),
            "owned_percentage": np.mean(owned_percentage_list),
            "discovered_percentage": np.mean(discovered_percentage_list),
            "disrupted_percentage": np.mean(disrupted_percentage_list),
            "number_steps": np.mean(number_steps_list),
            "network_availability": np.mean(network_availability_list),
            "episodes_won": np.mean(episode_won_list),
            "episodes_lost": np.mean(episode_lost_list),
            "reimaged": np.mean(reimaged_list),
            "reimaged_percentage": np.mean(reimaged_percentage_list),
            "num_events": np.mean(num_events_list),
            "min_distance_action": np.mean(min_distance_actions_list),
        }
        for i in range(len(outcome_actions_counts_list)):
            for outcome, count in outcome_actions_counts_list[i].items():
                stats[outcome + "_outcomes_count"] = stats.get(outcome + "_outcomes_count", 0) + count
        if outcome_actions_counts_list:
            for outcome, count in outcome_actions_counts_list[0].items():
                stats[outcome + "_outcomes_count"] = stats.get(outcome + "_outcomes_count", 0) / self.n_val_episodes

        overall_tags_list = []
        for tag_counts in target_node_tag_counts_list:
            for tag, count in tag_counts.items():
                if tag not in overall_tags_list:
                    overall_tags_list.append(tag)
                stats[tag + "_tags_count"] = stats.get(tag + "_tags_count", 0) + count

        for tag in overall_tags_list:
            stats[tag + "_tags_count"] = stats.get(tag + "_tags_count", 0) / self.n_val_episodes

        return stats

    def _log_tensorboard_custom_metrics(self, custom_metrics: Dict[str, float]) -> None:
        self.logger.record("validation/Episode reward (mean)", custom_metrics["episode_reward"])
        self.logger.record("validation/Owned nodes (mean)", custom_metrics["owned"])
        self.logger.record("validation/Discovered nodes (mean)", custom_metrics["discovered"])
        self.logger.record("validation/Discoverable amount (mean)", custom_metrics["discoverable_amount"])
        self.logger.record("validation/Disrupted nodes (mean)", custom_metrics["disrupted"])
        self.logger.record("validation/Not discovered nodes (mean)", custom_metrics["not_discovered"])
        self.logger.record("validation/Owned percentage (mean)", custom_metrics["owned_percentage"])
        self.logger.record("validation/Discovered percentage (mean)", custom_metrics["discovered_percentage"])
        self.logger.record("validation/Discovered amount percentage (mean)", custom_metrics["discovered_amount"])
        self.logger.record("validation/Disrupted percentage (mean)", custom_metrics["disrupted_percentage"])
        self.logger.record("validation/Owned-discovered ratio (mean)", custom_metrics["owned"] / custom_metrics["discovered"])
        self.logger.record("validation/Network availability (mean)", custom_metrics["network_availability"])
        self.logger.record("validation/Episodes won (mean)", custom_metrics["episodes_won"])
        self.logger.record("validation/Episodes lost (mean)", custom_metrics["episodes_lost"])
        self.logger.record("validation/Reimaged nodes (mean)", custom_metrics["reimaged"])
        self.logger.record("validation/Reimaged percentage (mean)", custom_metrics["reimaged_percentage"])
        self.logger.record("validation/Number of events", custom_metrics["num_events"])
        self.logger.record("validation/Minimum distance action (mean)", custom_metrics["min_distance_action"])

        self.logger.record("validation/Relative owned percentage (mean)", custom_metrics["owned"] / custom_metrics["reachable"])
        self.logger.record("validation/Relative discovered percentage (mean)", custom_metrics["discovered"] / custom_metrics["discoverable"])
        self.logger.record("validation/Relative disrupted percentage (mean)", custom_metrics["disrupted"] / custom_metrics["disruptable"])
        self.logger.record("validation/Relative discovered amount (mean)", custom_metrics["discovered_amount"] / custom_metrics["discoverable_amount"])

        self.logger.record("validation/Number of source nodes (mean)", custom_metrics["source_nodes"])
        self.logger.record("validation/Number of target nodes (mean)", custom_metrics["target_nodes"])
        self.logger.record("validation/Number of unique vulnerabilities (mean)", custom_metrics["actions_counts"])
        self.logger.record("validation/Number of steps (mean)", custom_metrics["number_steps"])

        self.logger.record("actions/validation/Invalid actions (mean)", custom_metrics["invalid_actions"])
        self.logger.record("actions/validation/Local actions count (mean)", custom_metrics["local_actions_count"])
        self.logger.record("actions/validation/Remote actions count (mean)", custom_metrics["remote_actions_count"])
        self.logger.record("actions/validation/Success rate for local actions (mean)", custom_metrics["local_actions_success"])
        self.logger.record("actions/validation/Success rate for remote actions (mean)", custom_metrics["remote_actions_success"])
        self.logger.record("actions/validation/Success rate for all actions (mean)", custom_metrics["actions_success"])

        for string, count in custom_metrics.items():
            if string.endswith("_outcomes_count"):
                outcome = string.split("_outcomes_count")[0]
                self.logger.record(f"actions/validation/Outcome {outcome} count (mean)", count)
            if string.endswith("_tags_count"):
                tag_name = string.split("_tags_count")[0]
                self.logger.record(f"validation/Target nodes {tag_name} count (mean)", count)
