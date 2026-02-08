from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cyberbattle.simulation import model

from .environment_event_source import EnvironmentEventSource, IEnvironmentObserver
from .reward_store import IRewardStore
from .learning_defender import LearningDefender
from .shared_env import unwrap_current_env


class DefenderEnvWrapper(gym.Env, IEnvironmentObserver):
    """Wraps a CyberBattleEnv for SB3 models to learn how to defend."""

    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]
    _log = logging.getLogger("cyberbattle.defender")
    _summary_log = logging.getLogger("marl.episode")

    def __init__(
        self,
        cyber_env: gym.Env,
        attacker_reward_store: IRewardStore,
        event_source: Optional[EnvironmentEventSource] = None,
        max_nodes: int = 1,
        max_total_services: int = 1,
        service_action_limit: int = 3,
        firewall_rule_list: Optional[Sequence[str]] = None,
        max_timesteps: int = 100,
        invalid_action_reward: float = 0.0,
        reset_on_constraint_broken: bool = True,
        loss_reward: float = -5000.0,
        defender_maintain_sla: float = 0.6,
        sla_worsening_penalty_scale: float = 200.0,
        availability_delta_scale: float = 0.0,
        owned_ratio_delta_scale: float = 0.0,
        attacker_reward_mode: str = "all",
        attacker_reward_scale: float = 1.0,
        candidate_k_hop: int = 1,
        candidate_random_nodes: int = 0,
        candidate_high_risk_nodes: int = 0,
        candidate_include_attack_target: bool = True,
        candidate_include_owned_neighbors: bool = True,
        log_episode_end: bool = False,
        log_episode_summary: bool = True,
        episode_log_prefix: str = "",
    ):
        super().__init__()
        self.cyber_env = cyber_env
        self.attacker_reward_store = attacker_reward_store
        self.max_nodes = int(max(1, max_nodes))
        self.max_total_services = int(max(1, max_total_services))
        self.service_action_limit = int(max(1, service_action_limit))
        if firewall_rule_list:
            self.firewall_rule_list = list(firewall_rule_list)
        self._firewall_port_to_idx = {port: idx for idx, port in enumerate(self.firewall_rule_list)}
        self.max_timesteps = int(max_timesteps)
        self.invalid_action_penalty = float(invalid_action_reward)
        self.reset_on_constraint_broken = bool(reset_on_constraint_broken)
        self.loss_reward = float(loss_reward)
        self.defender_maintain_sla = float(defender_maintain_sla)
        self.sla_worsening_penalty_scale = float(sla_worsening_penalty_scale)
        self.availability_delta_scale = float(availability_delta_scale)
        self.owned_ratio_delta_scale = float(owned_ratio_delta_scale)
        self.attacker_reward_mode = str(attacker_reward_mode).lower()
        self.attacker_reward_scale = float(attacker_reward_scale)
        self.candidate_k_hop = int(max(0, candidate_k_hop))
        self.candidate_random_nodes = int(max(0, candidate_random_nodes))
        self.candidate_high_risk_nodes = int(max(0, candidate_high_risk_nodes))
        self.candidate_include_attack_target = bool(candidate_include_attack_target)
        self.candidate_include_owned_neighbors = bool(candidate_include_owned_neighbors)

        self.defender = LearningDefender(cyber_env, firewall_rule_list=self.firewall_rule_list)

        self.action_space = self.__create_defender_action_space()
        self.observation_space = self.__create_observation_space()

        self.timesteps = 0
        self.rewards = []
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.action_count = 0
        self.last_action_type: Optional[int] = None
        self.reset_request = False
        self._has_breached_sla = False
        self._prev_network_availability = 1.0
        self._prev_owned_ratio = 0.0
        self.last_action = None
        self.last_action_valid = None
        self.last_reward = 0.0
        self.last_availability = 1.0
        self.last_worsening = 0.0
        self.last_sla_breached = False
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome: Optional[str] = None
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
        self.last_action_value: Optional[float] = None
        self.recent_action_values: Deque[float] = deque(maxlen=10)
        self._action_counts: Dict[str, int] = {}
        self._reward_component_sums: Dict[str, float] = {
            "attacker_component": 0.0,
            "invalid_penalty": 0.0,
            "sla_loss_penalty": 0.0,
            "sla_worsening_penalty": 0.0,
            "availability_shaping": 0.0,
            "owned_ratio_shaping": 0.0,
        }
        self.last_episode_reward_components: Optional[Dict[str, float]] = None
        self.recent_reward_components: Deque[Dict[str, float]] = deque(maxlen=10)
        self.sync_skip_count = 0
        self.last_episode_sync_skip_count: Optional[int] = None
        self.recent_sync_skip_counts: Deque[int] = deque(maxlen=10)

        if event_source is None:
            event_source = EnvironmentEventSource()
        self.event_source = event_source
        event_source.add_observer(self)

    def _get_attacker_outcome(self) -> Optional[model.VulnerabilityOutcome]:
        info = getattr(self.attacker_reward_store, "_last_info", None)
        if isinstance(info, dict) and "outcome_class" in info:
            return info["outcome_class"]
        return None

    def _attacker_outcome_defender_relevant(self, outcome: Optional[model.VulnerabilityOutcome]) -> bool:
        if outcome is None:
            return True
        if isinstance(
            outcome,
            (
                model.FirewallBlock,
                model.NonListeningPort,
                model.NonRunningMachine,
                model.Discovery,
                model.Reconnaissance,
                model.LateralMove,
                model.CredentialAccess,
                model.PrivilegeEscalation,
                model.Collection,
                model.Exfiltration,
                model.Persistence,
                model.DefenseEvasion,
                model.DenialOfService,
            ),
        ):
            return True
        if isinstance(
            outcome,
            (
                model.InvalidAction,
                model.NoVulnerability,
                model.NoEnoughPrivilege,
                model.OutcomeNonPresent,
                model.UnsuccessfulAction,
                model.NoNeededAction,
                model.RepeatedResult,
            ),
        ):
            return False
        return True

    def __create_observation_space(self) -> gym.Space:
        num_nodes = self.max_nodes
        fw_rules = len(self.firewall_rule_list)
        if fw_rules <= 0:
            fw_rules = 1
        return spaces.Dict(
            {
                "infected_nodes": spaces.MultiBinary(num_nodes),
                "incoming_firewall_status": spaces.MultiBinary(num_nodes * fw_rules),
                "outgoing_firewall_status": spaces.MultiBinary(num_nodes * fw_rules),
                "incoming_firewall_blocked": spaces.MultiBinary(num_nodes * fw_rules),
                "outgoing_firewall_blocked": spaces.MultiBinary(num_nodes * fw_rules),
                "node_status": spaces.MultiBinary(num_nodes * 3),
                "node_defense_evasion": spaces.MultiBinary(num_nodes),
                "node_persistence": spaces.MultiBinary(num_nodes),
                "node_reimageable": spaces.MultiBinary(num_nodes),
                "services_status": spaces.MultiBinary(self.max_total_services),
                "services_status_by_node": spaces.MultiBinary(num_nodes * self.service_action_limit),
                "services_available_by_node": spaces.MultiBinary(num_nodes * self.service_action_limit),
                "last_attack_target_node": spaces.MultiBinary(num_nodes),
                "last_attack_port": spaces.MultiBinary(fw_rules),
                "candidate_nodes": spaces.MultiBinary(num_nodes),
                "candidate_ports_by_node": spaces.MultiBinary(num_nodes * fw_rules),
                "candidate_services_by_node": spaces.MultiBinary(num_nodes * self.service_action_limit),
            }
        )

    def __create_defender_action_space(self) -> gym.Space:
        total_actions = 5
        fw_rules = len(self.firewall_rule_list)
        action_space = [
            total_actions,
            self.max_nodes,
            self.max_nodes,
            fw_rules,
            2,
            self.max_nodes,
            fw_rules,
            2,
            self.max_nodes,
            self.service_action_limit,
            self.max_nodes,
            self.service_action_limit,
        ]
        logging.info("Action space defender = %s", action_space)
        return spaces.MultiDiscrete(action_space)

    def _get_base_env(self):
        return unwrap_current_env(self.cyber_env)

    def on_reset(self, last_reward: float):
        self.reset_request = True

    def _infer_episode_reason(self, fallback_reason: Optional[str] = None) -> str:
        if fallback_reason:
            return fallback_reason
        if self.last_outcome:
            return str(self.last_outcome)
        base_env = None
        try:
            base_env = self._get_base_env()
        except Exception:
            base_env = None
        info_reason = None
        if isinstance(self.last_outcome, int):
            info_reason = self.last_outcome
        if base_env is not None and hasattr(base_env, "end_episode_reason"):
            try:
                info_reason = int(base_env.end_episode_reason)
            except Exception:
                info_reason = info_reason
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
        if self.last_truncated:
            return "truncated"
        if self.last_terminated:
            return "terminated"
        return "unknown"

    def _effective_firewall_permission(self, info, port_name: str, incoming: bool) -> model.RulePermission:
        rules = info.firewall.incoming if incoming else info.firewall.outgoing
        for rule in rules:
            if rule.port == port_name:
                return rule.permission
        return model.RulePermission.ALLOW

    def is_defender_action_valid(self, action: Sequence[int]) -> bool:
        base_env = self._get_base_env()
        environment = base_env.environment
        nodes = list(environment.nodes)

        def get_node(idx: int):
            if 0 <= idx < len(nodes):
                return nodes[idx]
            return None

        def node_info(node_id):
            return base_env.get_node(node_id)

        def node_running(info):
            return info.status == model.MachineStatus.Running

        def firewall_rule_exists(info, port_from_action: int, incoming: bool):
            if not (0 <= port_from_action < len(self.firewall_rule_list)):
                return False
            target_port = self.firewall_rule_list[port_from_action]
            rules = info.firewall.incoming if incoming else info.firewall.outgoing
            return any(rule.port == target_port for rule in rules)

        def service_exists(info, service_from_action: int):
            return 0 <= service_from_action < len(info.services)

        def service_running(info, service_from_action: int):
            if not service_exists(info, service_from_action):
                return False
            try:
                return bool(info.services[service_from_action].running)
            except Exception:
                return False

        def firewall_permission(info, port_from_action: int, incoming: bool):
            if not (0 <= port_from_action < len(self.firewall_rule_list)):
                return None
            target_port = self.firewall_rule_list[port_from_action]
            return self._effective_firewall_permission(info, target_port, incoming)

        action_number = int(action[0])
        if action_number == 0:
            node_id = get_node(int(action[1]))
            if node_id is None:
                return False
            info = node_info(node_id)
            return node_running(info) and info.reimageable
        if action_number == 1:
            node_id = get_node(int(action[2]))
            if node_id is None:
                return False
            info = node_info(node_id)
            if not node_running(info):
                return False
            perm = firewall_permission(info, int(action[3]), bool(action[4]))
            if perm is None:
                return False
            return perm != model.RulePermission.BLOCK
        if action_number == 2:
            node_id = get_node(int(action[5]))
            if node_id is None:
                return False
            info = node_info(node_id)
            if not node_running(info):
                return False
            perm = firewall_permission(info, int(action[6]), bool(action[7]))
            if perm is None:
                return False
            return perm == model.RulePermission.BLOCK
        if action_number == 3:
            node_id = get_node(int(action[8]))
            if node_id is None:
                return False
            info = node_info(node_id)
            return node_running(info) and service_running(info, int(action[9]))
        if action_number == 4:
            node_id = get_node(int(action[10]))
            if node_id is None:
                return False
            info = node_info(node_id)
            return node_running(info) and (not service_running(info, int(action[11])))
        return False

    def observe(self) -> Dict[str, np.ndarray]:
        base_env = self._get_base_env()
        environment = base_env.environment
        nodes = list(environment.nodes)
        num_nodes = len(nodes)
        fw_rules = len(self.firewall_rule_list)
        if fw_rules <= 0:
            fw_rules = 1

        infected = np.zeros((self.max_nodes,), dtype=np.int8)
        incoming_fw = np.zeros((self.max_nodes * fw_rules,), dtype=np.int8)
        outgoing_fw = np.zeros((self.max_nodes * fw_rules,), dtype=np.int8)
        incoming_fw_blocked = np.zeros((self.max_nodes * fw_rules,), dtype=np.int8)
        outgoing_fw_blocked = np.zeros((self.max_nodes * fw_rules,), dtype=np.int8)
        node_status = np.zeros((self.max_nodes * 3,), dtype=np.int8)
        node_defense_evasion = np.zeros((self.max_nodes,), dtype=np.int8)
        node_persistence = np.zeros((self.max_nodes,), dtype=np.int8)
        node_reimageable = np.zeros((self.max_nodes,), dtype=np.int8)
        services_status = np.zeros((self.max_total_services,), dtype=np.int8)
        services_status_by_node = np.zeros((self.max_nodes * self.service_action_limit,), dtype=np.int8)
        services_available_by_node = np.zeros((self.max_nodes * self.service_action_limit,), dtype=np.int8)
        last_attack_target_node = np.zeros((self.max_nodes,), dtype=np.int8)
        last_attack_port = np.zeros((fw_rules,), dtype=np.int8)
        candidate_nodes = np.zeros((self.max_nodes,), dtype=np.int8)
        candidate_ports_by_node = np.zeros((self.max_nodes * fw_rules,), dtype=np.int8)
        candidate_services_by_node = np.zeros((self.max_nodes * self.service_action_limit,), dtype=np.int8)

        try:
            target_node = getattr(base_env, "target_node", None)
            vuln_id = getattr(base_env, "vulnerability_ID", None)
            if target_node is not None and vuln_id is not None:
                if target_node in nodes:
                    node_idx = nodes.index(target_node)
                    if node_idx < self.max_nodes:
                        last_attack_target_node[node_idx] = 1
                try:
                    node_info = base_env.get_node(target_node)
                    if vuln_id in node_info.vulnerabilities:
                        port = node_info.vulnerabilities[vuln_id].port
                        port_idx = self._firewall_port_to_idx.get(port)
                        if port_idx is not None and 0 <= port_idx < fw_rules:
                            last_attack_port[port_idx] = 1
                except Exception:
                    pass
        except Exception:
            pass

        candidate_node_indices = set()
        if self.candidate_include_attack_target and last_attack_target_node.any():
            candidate_node_indices.update(np.where(last_attack_target_node[: self.max_nodes] > 0)[0].tolist())

        owned_node_ids = []
        try:
            owned_node_ids = list(getattr(base_env, "owned_nodes", []))
        except Exception:
            owned_node_ids = []

        if self.candidate_include_owned_neighbors and owned_node_ids and self.candidate_k_hop >= 0:
            node_index_map = {node_id: idx for idx, node_id in enumerate(nodes)}
            seed_ids = [node_id for node_id in owned_node_ids if node_id in node_index_map]
            for seed_id in seed_ids:
                seed_idx = node_index_map[seed_id]
                if seed_idx < self.max_nodes:
                    candidate_node_indices.add(seed_idx)
                if self.candidate_k_hop <= 0:
                    continue
                frontier = {seed_id}
                visited = {seed_id}
                for _ in range(self.candidate_k_hop):
                    next_frontier = set()
                    for current in frontier:
                        for neighbor in environment.neighbors(current):
                            if neighbor in visited:
                                continue
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                            if neighbor in node_index_map:
                                neighbor_idx = node_index_map[neighbor]
                                if neighbor_idx < self.max_nodes:
                                    candidate_node_indices.add(neighbor_idx)
                    frontier = next_frontier
                    if not frontier:
                        break

        if self.candidate_high_risk_nodes > 0:
            risk_scores = []
            for node_idx, node_id in enumerate(nodes[: self.max_nodes]):
                node_info = base_env.get_node(node_id)
                vuln_count = len(getattr(node_info, "vulnerabilities", {}))
                svc_count = len(getattr(node_info, "services", []))
                risk_scores.append((vuln_count + svc_count, node_idx))
            for _, node_idx in sorted(risk_scores, reverse=True)[: self.candidate_high_risk_nodes]:
                candidate_node_indices.add(node_idx)

        if self.candidate_random_nodes > 0 and num_nodes > 0:
            random_choices = np.random.choice(
                min(self.max_nodes, num_nodes),
                size=min(self.candidate_random_nodes, min(self.max_nodes, num_nodes)),
                replace=False,
            )
            candidate_node_indices.update(int(i) for i in random_choices)

        if not candidate_node_indices:
            candidate_node_indices = set(range(min(self.max_nodes, num_nodes)))

        for idx in candidate_node_indices:
            if 0 <= idx < self.max_nodes:
                candidate_nodes[idx] = 1

        service_idx = 0
        for node_idx, node_id in enumerate(nodes[: self.max_nodes]):
            node_info = base_env.get_node(node_id)
            infected[node_idx] = 1 if node_info.agent_installed else 0
            node_defense_evasion[node_idx] = 1 if getattr(node_info, "defense_evasion", False) else 0
            node_persistence[node_idx] = 1 if getattr(node_info, "persistence", False) else 0
            node_reimageable[node_idx] = 1 if getattr(node_info, "reimageable", False) else 0

            status_offset = node_idx * 3
            if node_info.status == model.MachineStatus.Running:
                node_status[status_offset] = 1
            elif node_info.status == model.MachineStatus.Stopped:
                node_status[status_offset + 1] = 1
            elif node_info.status == model.MachineStatus.Imaging:
                node_status[status_offset + 2] = 1

            for rule_idx, rule in enumerate(self.firewall_rule_list):
                incoming_active = any(r.port == rule for r in node_info.firewall.incoming)
                outgoing_active = any(r.port == rule for r in node_info.firewall.outgoing)
                incoming_fw[node_idx * fw_rules + rule_idx] = 1 if incoming_active else 0
                outgoing_fw[node_idx * fw_rules + rule_idx] = 1 if outgoing_active else 0
                incoming_perm = self._effective_firewall_permission(node_info, rule, True)
                outgoing_perm = self._effective_firewall_permission(node_info, rule, False)
                incoming_fw_blocked[node_idx * fw_rules + rule_idx] = 1 if incoming_perm == model.RulePermission.BLOCK else 0
                outgoing_fw_blocked[node_idx * fw_rules + rule_idx] = 1 if outgoing_perm == model.RulePermission.BLOCK else 0

            for service_idx_local, service in enumerate(node_info.services):
                if service_idx >= self.max_total_services:
                    break
                services_status[service_idx] = 1 if service.running else 0
                service_idx += 1
                if service_idx_local < self.service_action_limit:
                    services_available_by_node[node_idx * self.service_action_limit + service_idx_local] = 1
                    services_status_by_node[node_idx * self.service_action_limit + service_idx_local] = 1 if service.running else 0

            if candidate_nodes[node_idx] > 0:
                candidate_services_by_node[
                    node_idx * self.service_action_limit : (node_idx + 1) * self.service_action_limit
                ] = services_available_by_node[
                    node_idx * self.service_action_limit : (node_idx + 1) * self.service_action_limit
                ]
                port_candidates = set()
                try:
                    for vuln in node_info.vulnerabilities.values():
                        port_candidates.add(vuln.port)
                except Exception:
                    pass
                try:
                    for service in node_info.services:
                        port_candidates.add(service.name)
                except Exception:
                    pass
                try:
                    for rule in node_info.firewall.incoming:
                        port_candidates.add(rule.port)
                    for rule in node_info.firewall.outgoing:
                        port_candidates.add(rule.port)
                except Exception:
                    pass
                if last_attack_target_node[node_idx] and last_attack_port.any():
                    for port_idx in np.where(last_attack_port > 0)[0]:
                        if 0 <= port_idx < len(self.firewall_rule_list):
                            port_candidates.add(self.firewall_rule_list[port_idx])
                if port_candidates:
                    for port in port_candidates:
                        port_idx = self._firewall_port_to_idx.get(port)
                        if port_idx is not None:
                            candidate_ports_by_node[node_idx * fw_rules + port_idx] = 1
                else:
                    candidate_ports_by_node[
                        node_idx * fw_rules : (node_idx + 1) * fw_rules
                    ] = 1

        return {
            "infected_nodes": infected,
            "incoming_firewall_status": incoming_fw,
            "outgoing_firewall_status": outgoing_fw,
            "incoming_firewall_blocked": incoming_fw_blocked,
            "outgoing_firewall_blocked": outgoing_fw_blocked,
            "node_status": node_status,
            "node_defense_evasion": node_defense_evasion,
            "node_persistence": node_persistence,
            "node_reimageable": node_reimageable,
            "services_status": services_status,
            "services_status_by_node": services_status_by_node,
            "services_available_by_node": services_available_by_node,
            "last_attack_target_node": last_attack_target_node,
            "last_attack_port": last_attack_port,
            "candidate_nodes": candidate_nodes,
            "candidate_ports_by_node": candidate_ports_by_node,
            "candidate_services_by_node": candidate_services_by_node,
        }

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0.0
        invalid_penalty = 0.0
        attacker_component = 0.0
        sla_loss_penalty = 0.0
        sla_worsening_penalty = 0.0
        availability_shaping = 0.0
        owned_ratio_shaping = 0.0
        used_action = action

        pending = False
        pending_reason = None
        if hasattr(self.cyber_env, "consume_episode_done"):
            result = self.cyber_env.consume_episode_done()
            if isinstance(result, tuple):
                pending, pending_reason = result
            else:
                pending = bool(result)

        if pending:
            defender_observation = self.observe()
            truncated = True
            reward = 0.0
            self.sync_skip_count += 1
            self.rewards.append(reward)
            self.timesteps += 1
            done = True
            if done and not self._episode_end_recorded:
                total_reward = float(np.sum(self.rewards)) if self.rewards else 0.0
                self.last_episode_reward = total_reward
                self.last_episode_len = int(self.timesteps)
                self.recent_episode_rewards.append(total_reward)
                self.recent_episode_lens.append(int(self.timesteps))
                self.last_episode_action_count = int(self.action_count)
                self.recent_action_counts.append(int(self.action_count))
                self.last_invalid_action_count = int(self.invalid_action_count)
                self.recent_invalid_action_counts.append(int(self.invalid_action_count))
                self.last_episode_reward_components = dict(self._reward_component_sums)
                self.recent_reward_components.append(dict(self._reward_component_sums))
                self.last_episode_sync_skip_count = int(self.sync_skip_count)
                self.recent_sync_skip_counts.append(int(self.sync_skip_count))
                self._episode_end_recorded = True
            if done and self._log_episode_summary and not self._episode_summary_logged:
                actions_sorted = dict(sorted(self._action_counts.items(), key=lambda kv: (-kv[1], kv[0])))
                reason = self._infer_episode_reason(pending_reason)
                comp = self.last_episode_reward_components or dict(self._reward_component_sums)
                sync_skip_ratio = (float(self.sync_skip_count) / float(self.timesteps)) if self.timesteps else 0.0
                self._summary_log.info(
                    "DEFENDER EPISODE SUMMARY reward=%.3f len=%s reason=%s invalid_actions=%d sync_skip=%d sync_skip_ratio=%.3f reward_components=%s actions=%s",
                    float(self.last_episode_reward or 0.0),
                    str(self.last_episode_len),
                    reason,
                    int(self.last_invalid_action_count or 0),
                    int(self.sync_skip_count),
                    float(sync_skip_ratio),
                    comp,
                    actions_sorted,
                )
                self._episode_summary_logged = True
            return defender_observation, float(reward), False, True, {"sync_skip": True}

        if action is None or (hasattr(action, "__len__") and len(action) == 0):
            self.last_action = np.array([], dtype=int)
            action_valid = True
            self.last_action_valid = True
            used_action = []
        else:
            self.last_action = np.array(action, copy=True) if hasattr(action, "__len__") else action
            action_valid = bool(self.is_defender_action_valid(action))
        self.last_action_valid = action_valid
        try:
            self.last_action_type = int(self.last_action[0]) if self.last_action is not None and len(self.last_action) > 0 else None
        except Exception:
            self.last_action_type = None
        if self.last_action_type is not None:
            self.last_action_value = float(self.last_action_type)
        else:
            self.last_action_value = None

        action_label = "NOOP"
        if self.last_action is not None and hasattr(self.last_action, "__len__") and len(self.last_action) > 0:
            action_map = {
                0: "REIMAGE",
                1: "BLOCK_TRAFFIC",
                2: "ALLOW_TRAFFIC",
                3: "STOP_SERVICE",
                4: "START_SERVICE",
            }
            action_label = action_map.get(self.last_action_type, "UNKNOWN")
        if not action_valid:
            action_label = "INVALID"
        self._action_counts[action_label] = self._action_counts.get(action_label, 0) + 1

        if not action_valid:
            self.invalid_action_count += 1
            invalid_penalty = float(self.invalid_action_penalty)
            reward += invalid_penalty
            used_action = []
        else:
            self.valid_action_count += 1

        self.defender.executeAction(used_action)
        self.action_count += 1

        base_env = self._get_base_env()
        actuator = base_env._defender_actuator  # noqa: SLF001
        current_availability = float(actuator.network_availability)
        self.last_availability = current_availability
        self.last_sla_breached = bool(current_availability < self.defender_maintain_sla)
        worsening = float(self._prev_network_availability - current_availability)
        self.last_worsening = float(max(0.0, worsening))
        if self.availability_delta_scale:
            availability_shaping = float(self.availability_delta_scale) * float(current_availability - self._prev_network_availability)
            reward += availability_shaping

        if self.owned_ratio_delta_scale:
            try:
                total_nodes = getattr(base_env, "num_nodes", None)
                if not total_nodes:
                    total_nodes = len(list(base_env.environment.nodes))
                owned_ratio = float(len(getattr(base_env, "owned_nodes", []))) / float(total_nodes) if total_nodes else 0.0
                owned_ratio_shaping = float(self.owned_ratio_delta_scale) * float(self._prev_owned_ratio - owned_ratio)
                reward += owned_ratio_shaping
                self._prev_owned_ratio = owned_ratio
            except Exception:
                owned_ratio_shaping = 0.0

        if self.attacker_reward_store.episode_rewards:
            attacker_component = float(-1.0 * self.attacker_reward_store.episode_rewards[-1])
            if self.attacker_reward_mode == "defender_only":
                outcome = self._get_attacker_outcome()
                if not self._attacker_outcome_defender_relevant(outcome):
                    attacker_component = 0.0
            attacker_component *= float(self.attacker_reward_scale)
            reward += attacker_component

        if current_availability < self.defender_maintain_sla:
            if not self._has_breached_sla:
                sla_loss_penalty = float(self.loss_reward)
                reward += sla_loss_penalty
                if self.reset_on_constraint_broken:
                    terminated = True
                    self.last_outcome = "sla_breached"
                self._has_breached_sla = True
            else:
                if worsening > 0:
                    sla_worsening_penalty = float(-self.sla_worsening_penalty_scale * worsening)
                    reward += sla_worsening_penalty
        else:
            self._has_breached_sla = False

        self._prev_network_availability = current_availability

        if getattr(base_env, "defender_goal_reached", None) and base_env.defender_goal_reached():
            reward = float(getattr(base_env, "winning_reward", reward))
            terminated = True
            self.last_outcome = "defender_win"

        defender_observation = self.observe()
        self.timesteps += 1

        if self.reset_request:
            truncated = True
            reward = -1.0 * float(self.attacker_reward_store.episode_rewards[-1]) if self.attacker_reward_store.episode_rewards else reward
        elif self.timesteps >= self.max_timesteps:
            truncated = True

        if self.last_outcome is None and truncated and self.timesteps >= self.max_timesteps:
            self.last_outcome = "timeout"

        self._reward_component_sums["attacker_component"] += float(attacker_component)
        self._reward_component_sums["invalid_penalty"] += float(invalid_penalty)
        self._reward_component_sums["sla_loss_penalty"] += float(sla_loss_penalty)
        self._reward_component_sums["sla_worsening_penalty"] += float(sla_worsening_penalty)
        self._reward_component_sums["availability_shaping"] += float(availability_shaping)
        self._reward_component_sums["owned_ratio_shaping"] += float(owned_ratio_shaping)

        self.rewards.append(reward)
        done = bool(terminated or truncated)
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)

        if done and not self._episode_end_recorded:
            total_reward = float(np.sum(self.rewards)) if self.rewards else 0.0
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
            self.last_episode_reward_components = dict(self._reward_component_sums)
            self.recent_reward_components.append(dict(self._reward_component_sums))
            self.last_episode_sync_skip_count = int(self.sync_skip_count)
            self.recent_sync_skip_counts.append(int(self.sync_skip_count))
            self._episode_end_recorded = True

        if done and self._log_episode_summary and not self._episode_summary_logged:
            actions_sorted = dict(sorted(self._action_counts.items(), key=lambda kv: (-kv[1], kv[0])))
            reason = self._infer_episode_reason()
            comp = self.last_episode_reward_components or dict(self._reward_component_sums)
            sync_skip_ratio = (float(self.sync_skip_count) / float(self.timesteps)) if self.timesteps else 0.0
            self._summary_log.info(
                "DEFENDER EPISODE SUMMARY reward=%.3f len=%s reason=%s invalid_actions=%d sync_skip=%d sync_skip_ratio=%.3f reward_components=%s actions=%s",
                float(self.last_episode_reward or 0.0),
                str(self.last_episode_len),
                reason,
                int(self.last_invalid_action_count or 0),
                int(self.sync_skip_count),
                float(sync_skip_ratio),
                comp,
                actions_sorted,
            )
            self._episode_summary_logged = True

        if done and hasattr(self.cyber_env, "signal_episode_done"):
            self.cyber_env.signal_episode_done(self._infer_episode_reason())

        if self._log_episode_end and done and not self._episode_end_logged:
            if self.last_outcome:
                reason = self.last_outcome
            elif self.last_truncated:
                reason = "truncated"
            elif self.last_terminated:
                reason = "terminated"
            elif self.timesteps >= self.max_timesteps:
                reason = "timeout"
            else:
                reason = "unknown"
            prefix = f"{self._episode_log_prefix} " if self._episode_log_prefix else ""
            total_reward = float(np.sum(self.rewards)) if self.rewards else 0.0
            self._log.warning(
                "%sDEFENDER EPISODE END steps=%d reason=%s total_reward=%.3f valid=%d invalid=%d availability=%.3f sla_breached=%s",
                prefix,
                int(self.timesteps),
                reason,
                total_reward,
                int(self.valid_action_count),
                int(self.invalid_action_count),
                float(self.last_availability),
                bool(self.last_sla_breached),
            )
            self._episode_end_logged = True

        return defender_observation, float(reward), bool(terminated), bool(truncated), {}

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict[str, Any]]:
        if not self.reset_request:
            self.event_source.notify_reset(last_reward=0)

        if (self.timesteps is not None and self.timesteps > 0 and not self._episode_end_recorded):
            total_reward = float(np.sum(self.rewards)) if self.rewards else 0.0
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
            self._episode_end_recorded = True

        reset_result = self.cyber_env.reset(seed=seed, options=options)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            _, info = reset_result
        else:
            info = {}

        self.reset_request = False
        self.rewards = []
        self.timesteps = 0
        self.valid_action_count = 0
        self.invalid_action_count = 0
        self.action_count = 0
        self._has_breached_sla = False
        base_env = self._get_base_env()
        actuator = base_env._defender_actuator  # noqa: SLF001
        self._prev_network_availability = float(actuator.network_availability)
        self.last_action = None
        self.last_action_valid = None
        self.last_action_type = None
        self.last_action_value = None
        self.last_reward = 0.0
        self.last_availability = float(actuator.network_availability)
        self.last_worsening = 0.0
        self.last_sla_breached = False
        self.last_terminated = False
        self.last_truncated = False
        self.last_outcome = None
        self._episode_end_logged = False
        self._episode_end_recorded = False
        self._episode_summary_logged = False
        self._action_counts = {}
        self._reward_component_sums = {
            "attacker_component": 0.0,
            "invalid_penalty": 0.0,
            "sla_loss_penalty": 0.0,
            "sla_worsening_penalty": 0.0,
            "availability_shaping": 0.0,
            "owned_ratio_shaping": 0.0,
        }
        self.last_episode_reward_components = None
        self.sync_skip_count = 0
        self._prev_owned_ratio = 0.0

        return self.observe(), dict(info) if isinstance(info, dict) else {}

    def close(self):
        return self.cyber_env.close()

    def render(self, *args, **kwargs):
        return self.cyber_env.render(*args, **kwargs)
