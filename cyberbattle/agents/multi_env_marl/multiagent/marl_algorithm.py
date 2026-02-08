"""Multi-agent training loop (attacker + defender)."""
from __future__ import annotations

from typing import Optional

from .baseline_agent import BaselineMultiAgent


def collect_rollouts_multi(
    attacker_agent: BaselineMultiAgent,
    defender_agent: BaselineMultiAgent,
    defender_first: bool = False,
) -> bool:
    attacker_agent.on_rollout_start()
    defender_agent.on_rollout_start()

    n_steps = 0
    max_steps = min(attacker_agent.n_rollout_steps, defender_agent.n_rollout_steps)

    new_obs1 = None
    new_obs2 = None
    dones1 = None
    dones2 = None

    while n_steps < max_steps:
        if defender_first:
            continue2, new_obs2, dones2 = defender_agent.perform_step(n_steps)
            continue1, new_obs1, dones1 = attacker_agent.perform_step(n_steps)
        else:
            continue1, new_obs1, dones1 = attacker_agent.perform_step(n_steps)
            continue2, new_obs2, dones2 = defender_agent.perform_step(n_steps)
        if continue1 is False or continue2 is False:
            return False
        n_steps += 1

    attacker_agent.on_rollout_end(new_obs1, dones1)
    defender_agent.on_rollout_end(new_obs2, dones2)
    return True


def learn_multi(
    attacker_agent: BaselineMultiAgent,
    defender_agent: BaselineMultiAgent,
    total_timesteps: int,
    checkpoint_callback: Optional[callable] = None,
    checkpoint_interval: int = 0,
    reset_num_timesteps: bool = True,
    defender_first: bool = False,
) -> int:
    iteration = 0

    attacker_total = int(total_timesteps)
    defender_total = int(total_timesteps)

    attacker_agent.setup_learn(attacker_total, None, -1, 0, None, reset_num_timesteps, "OnPolicyAlgorithm")
    defender_agent.setup_learn(defender_total, None, -1, 0, None, reset_num_timesteps, "OnPolicyAlgorithm")

    last_checkpoint_iteration = None

    while attacker_agent.num_timesteps < attacker_total and defender_agent.num_timesteps < defender_total:
        continue_training = collect_rollouts_multi(attacker_agent, defender_agent, defender_first=defender_first)
        if not continue_training:
            break

        iteration += 1

        attacker_agent.update_progress(attacker_total)
        defender_agent.update_progress(defender_total)

        if iteration % attacker_agent.log_interval == 0:
            attacker_agent.log_training(iteration)
        if iteration % defender_agent.log_interval == 0:
            defender_agent.log_training(iteration)

        attacker_agent.train()
        defender_agent.train()

        if checkpoint_callback and checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            checkpoint_callback(iteration)
            last_checkpoint_iteration = iteration

    attacker_agent.on_training_end()
    defender_agent.on_training_end()

    if checkpoint_callback and last_checkpoint_iteration != iteration:
        checkpoint_callback(iteration)

    return iteration
