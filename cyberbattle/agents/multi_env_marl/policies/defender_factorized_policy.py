from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


def _safe_reshape(x: torch.Tensor, batch: int, dim: int) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros((batch, dim))
    return x.view(batch, dim)


class FactorizedDefenderPolicy(ActorCriticPolicy):
    """
    Defender policy that learns node attention first, then emits action logits
    conditioned on the attended node embedding.

    This is a factorized/autoregressive-style policy without hard-coded rules:
    - node logits are learned from per-node features
    - action-type/port/service logits are conditioned on an attended node vector
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule, **kwargs):
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise TypeError("FactorizedDefenderPolicy requires MultiDiscrete action space")
        self._mask_mode = str(kwargs.pop("candidate_mask_mode", "soft")).lower()
        self._node_prior_strength = float(kwargs.pop("node_prior_strength", 2.0))
        self._port_prior_strength = float(kwargs.pop("port_prior_strength", 2.0))
        self._service_prior_strength = float(kwargs.pop("service_prior_strength", 1.0))
        self._condition_on_sampled_node = bool(kwargs.pop("condition_on_sampled_node", True))
        self._condition_on_action_node = bool(kwargs.pop("condition_on_action_node", True))
        self._canonical_node_index = int(kwargs.pop("canonical_node_index", 1))
        self._enforce_node_consistency = bool(kwargs.pop("enforce_node_consistency", True))
        self._action_nvec = action_space.nvec.tolist()
        if len(self._action_nvec) != 12:
            raise ValueError(f"Unexpected defender action space size: {self._action_nvec}")
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Unpack action dimensions
        (
            self._n_action_types,
            self._n_nodes,
            _,  # block_node
            self._n_ports,
            self._n_dirs,
            _,  # allow_node
            _,  # allow_port
            _,  # allow_dir
            _,  # stop_node
            self._n_services,
            _,  # start_node
            _,  # start_service
        ) = self._action_nvec

        # Node feature sizes derived from observation layout
        obs_spaces = observation_space.spaces
        self._n_nodes = int(self._n_nodes)
        self._n_ports = int(self._n_ports)
        self._n_services = int(self._n_services)

        # Model sizes
        node_feat_dim = 0
        if "infected_nodes" in obs_spaces:
            node_feat_dim += 1
        if "node_status" in obs_spaces:
            node_feat_dim += 3
        if "node_defense_evasion" in obs_spaces:
            node_feat_dim += 1
        if "node_persistence" in obs_spaces:
            node_feat_dim += 1
        if "node_reimageable" in obs_spaces:
            node_feat_dim += 1
        if "services_status_by_node" in obs_spaces:
            node_feat_dim += 1
        if "services_available_by_node" in obs_spaces:
            node_feat_dim += 1
        if "incoming_firewall_blocked" in obs_spaces:
            node_feat_dim += 1
        if "outgoing_firewall_blocked" in obs_spaces:
            node_feat_dim += 1
        if "last_attack_target_node" in obs_spaces:
            node_feat_dim += 1

        hidden = 128
        self._node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self._node_attention = nn.Linear(hidden, 1)
        self._action_type_head = nn.Linear(hidden, self._n_action_types)
        self._port_head = nn.Linear(hidden, self._n_ports)
        self._dir_head = nn.Linear(hidden, self._n_dirs)
        self._service_head = nn.Linear(hidden, self._n_services)
        self._value_head = nn.Linear(hidden, 1)

        self._action_dist = MultiCategoricalDistribution(self._action_nvec)
        self._node_action_indices = [1, 2, 5, 8, 10]

        self._init_weights()

    def _init_weights(self) -> None:
        for module in (
            self._node_encoder,
            self._node_attention,
            self._action_type_head,
            self._port_head,
            self._dir_head,
            self._service_head,
            self._value_head,
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def _build_node_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = next(iter(obs.values())).shape[0]
        features = []

        def add_feature(name: str, size: int) -> None:
            if name not in obs:
                return
            tensor = obs[name].float()
            features.append(_safe_reshape(tensor, batch, self._n_nodes * size).view(batch, self._n_nodes, size))

        add_feature("infected_nodes", 1)
        add_feature("node_status", 3)
        add_feature("node_defense_evasion", 1)
        add_feature("node_persistence", 1)
        add_feature("node_reimageable", 1)

        if "services_status_by_node" in obs:
            svc = obs["services_status_by_node"].float().view(batch, self._n_nodes, -1)
            svc_ratio = svc.mean(dim=-1, keepdim=True)
            features.append(svc_ratio)
        if "services_available_by_node" in obs:
            svc = obs["services_available_by_node"].float().view(batch, self._n_nodes, -1)
            svc_ratio = svc.mean(dim=-1, keepdim=True)
            features.append(svc_ratio)
        if "incoming_firewall_blocked" in obs:
            blocked = obs["incoming_firewall_blocked"].float().view(batch, self._n_nodes, -1)
            blocked_ratio = blocked.mean(dim=-1, keepdim=True)
            features.append(blocked_ratio)
        if "outgoing_firewall_blocked" in obs:
            blocked = obs["outgoing_firewall_blocked"].float().view(batch, self._n_nodes, -1)
            blocked_ratio = blocked.mean(dim=-1, keepdim=True)
            features.append(blocked_ratio)
        if "last_attack_target_node" in obs:
            lat = obs["last_attack_target_node"].float().view(batch, self._n_nodes, 1)
            features.append(lat)

        if not features:
            return torch.zeros((batch, self._n_nodes, 1), device=next(iter(obs.values())).device)
        return torch.cat(features, dim=-1)

    def _compute_logits_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        selected_nodes: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self._build_node_features(obs)
        node_embeddings = self._node_encoder(node_features)
        node_logits = self._node_attention(node_embeddings).squeeze(-1)
        if "candidate_nodes" in obs:
            node_mask = obs["candidate_nodes"].float().view(-1, self._n_nodes)
            if torch.any(node_mask > 0):
                if self._mask_mode == "hard":
                    node_logits = node_logits.masked_fill(node_mask <= 0, -1e9)
                else:
                    node_bias = (node_mask * 2.0 - 1.0) * self._node_prior_strength
                    node_logits = node_logits + node_bias
        node_weights = torch.softmax(node_logits, dim=-1).unsqueeze(-1)
        weighted_context = torch.sum(node_embeddings * node_weights, dim=1)
        context = weighted_context
        batch_indices = None
        if selected_nodes is not None:
            selected_nodes = selected_nodes.long().clamp(0, self._n_nodes - 1)
            batch_indices = torch.arange(selected_nodes.shape[0], device=selected_nodes.device)
            context = node_embeddings[batch_indices, selected_nodes]

        action_type_logits = self._action_type_head(context)
        port_logits = self._port_head(context)
        dir_logits = self._dir_head(context)
        service_logits = self._service_head(context)
        value = self._value_head(weighted_context).squeeze(-1)

        if "candidate_ports_by_node" in obs:
            port_mask = obs["candidate_ports_by_node"].float().view(-1, self._n_nodes, self._n_ports)
            if torch.any(port_mask > 0):
                if batch_indices is not None:
                    effective_port_mask = port_mask[batch_indices, selected_nodes]
                else:
                    effective_port_mask = torch.sum(port_mask * node_weights, dim=1)
                if self._mask_mode == "hard":
                    port_logits = port_logits.masked_fill(effective_port_mask <= 0, -1e9)
                else:
                    port_bias = (effective_port_mask * 2.0 - 1.0) * self._port_prior_strength
                    port_logits = port_logits + port_bias

        if "candidate_services_by_node" in obs:
            service_mask = obs["candidate_services_by_node"].float().view(-1, self._n_nodes, self._n_services)
            if torch.any(service_mask > 0):
                if batch_indices is not None:
                    effective_service_mask = service_mask[batch_indices, selected_nodes]
                else:
                    effective_service_mask = torch.sum(service_mask * node_weights, dim=1)
                if self._mask_mode == "hard":
                    service_logits = service_logits.masked_fill(effective_service_mask <= 0, -1e9)
                else:
                    service_bias = (effective_service_mask * 2.0 - 1.0) * self._service_prior_strength
                    service_logits = service_logits + service_bias

        logits = torch.cat(
            [
                action_type_logits,
                node_logits,
                node_logits,
                port_logits,
                dir_logits,
                node_logits,
                port_logits,
                dir_logits,
                node_logits,
                service_logits,
                node_logits,
                service_logits,
            ],
            dim=1,
        )
        return logits, value, node_logits

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        selected_nodes = None
        logits, value, node_logits = self._compute_logits_and_value(obs)
        if self._condition_on_sampled_node:
            node_dist = torch.distributions.Categorical(logits=node_logits)
            selected_nodes = node_dist.probs.argmax(dim=-1) if deterministic else node_dist.sample()
            logits, value, _ = self._compute_logits_and_value(obs, selected_nodes=selected_nodes)
        dist = self._action_dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        if self._enforce_node_consistency and selected_nodes is not None:
            actions = actions.clone()
            for idx in self._node_action_indices:
                actions[:, idx] = selected_nodes
        log_prob = dist.log_prob(actions)
        return actions, value, log_prob

    def _get_action_dist_from_obs(self, obs: Dict[str, torch.Tensor]) -> MultiCategoricalDistribution:
        logits, _, _ = self._compute_logits_and_value(obs)
        return self._action_dist.proba_distribution(action_logits=logits)

    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> MultiCategoricalDistribution:
        return self._get_action_dist_from_obs(obs)

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        selected_nodes = None
        if self._condition_on_action_node and actions is not None and actions.ndim == 2 and actions.shape[1] > self._canonical_node_index:
            selected_nodes = actions[:, self._canonical_node_index]
        logits, value, _ = self._compute_logits_and_value(obs, selected_nodes=selected_nodes)
        dist = self._action_dist.proba_distribution(action_logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, value, _ = self._compute_logits_and_value(obs)
        return value
