from __future__ import annotations

import logging
from typing import Sequence

from cyberbattle.simulation import model

from .shared_env import unwrap_current_env


class LearningDefender:
    """Defender executor that applies actions via StaticDefenderAgentActions."""

    firewall_rule_list = ["RDP", "SSH", "HTTPS", "HTTP", "su", "sudo"]

    def __init__(self, cyber_env, firewall_rule_list=None):
        self.cyber_env = cyber_env
        if firewall_rule_list:
            self.firewall_rule_list = list(firewall_rule_list)

    def _get_base_env(self):
        return unwrap_current_env(self.cyber_env)

    def executeAction(self, next_action: Sequence[int]) -> None:
        base_env = self._get_base_env()
        actuator = base_env._defender_actuator  # noqa: SLF001
        environment = base_env.environment

        actuator.on_attacker_step_taken()

        if next_action is None:
            return
        try:
            if hasattr(next_action, "__len__") and len(next_action) == 0:
                return
        except Exception:
            return

        try:
            action_number = int(next_action[0])
        except Exception:
            return

        nodes = list(environment.nodes)

        def get_node_from_action(idx: int):
            if 0 <= idx < len(nodes):
                return nodes[idx]
            return None

        def get_firewall_port_name(idx: int):
            if 0 <= idx < len(self.firewall_rule_list):
                return self.firewall_rule_list[idx]
            return None

        if action_number == 0:
            node_id = get_node_from_action(int(next_action[1]))
            if node_id is not None:
                actuator.reimage_node(node_id)
            return

        if action_number == 1:
            node_id = get_node_from_action(int(next_action[2]))
            port_name = get_firewall_port_name(int(next_action[3]))
            incoming = bool(next_action[4])
            if node_id is not None and port_name is not None:
                actuator.block_traffic(node_id, port_name, incoming)
            return

        if action_number == 2:
            node_id = get_node_from_action(int(next_action[5]))
            port_name = get_firewall_port_name(int(next_action[6]))
            incoming = bool(next_action[7])
            if node_id is not None and port_name is not None:
                actuator.allow_traffic(node_id, port_name, incoming)
            return

        if action_number == 3:
            node_id = get_node_from_action(int(next_action[8]))
            service_idx = int(next_action[9])
            if node_id is None:
                return
            node_info = environment.nodes(data=True)[node_id]["data"]
            if node_info.status != model.MachineStatus.Running:
                logging.debug("Skip stop_service on non-running node %s", node_id)
                return
            if 0 <= service_idx < len(node_info.services):
                actuator.stop_service(node_id, node_info.services[service_idx].name)
            return

        if action_number == 4:
            node_id = get_node_from_action(int(next_action[10]))
            service_idx = int(next_action[11])
            if node_id is None:
                return
            node_info = environment.nodes(data=True)[node_id]["data"]
            if node_info.status != model.MachineStatus.Running:
                logging.debug("Skip start_service on non-running node %s", node_id)
                return
            if 0 <= service_idx < len(node_info.services):
                actuator.start_service(node_id, node_info.services[service_idx].name)
            return
