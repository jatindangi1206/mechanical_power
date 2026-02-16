"""
MDP dataset construction: convert preprocessed patient time-series into
(state, action, reward, next_state, terminal) tuples for offline RL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold


# ===================================================================
# Action discretisation
# ===================================================================
def discretise_action(mp_delta: float, action_bins: dict) -> int:
    """
    Map a continuous MP change to one of the discrete action indices.

    action_bins example from config:
        {0: {"delta": -5}, 1: {"delta": -2}, 2: {"delta": 0},
         3: {"delta": +2}, 4: {"delta": +5}}
    """
    thresholds = sorted(
        [(int(k), v["delta"]) for k, v in action_bins.items()],
        key=lambda x: x[1],
    )
    best_action = thresholds[len(thresholds) // 2][0]  # default: maintain
    best_dist = float("inf")
    for action_idx, delta in thresholds:
        dist = abs(mp_delta - delta)
        if dist < best_dist:
            best_dist = dist
            best_action = action_idx
    return best_action


# ===================================================================
# Reward calculation
# ===================================================================
def calculate_reward(
    state_t: dict,
    state_t1: dict,
    is_terminal: bool,
    outcome: str | None,
    reward_cfg: dict,
) -> float:
    """
    Multi-objective reward function encoding clinical priorities.

    Parameters
    ----------
    state_t   : feature dict at time t
    state_t1  : feature dict at time t+1
    is_terminal : whether t+1 is the last step
    outcome   : 'survived' or 'died' (only meaningful if terminal)
    reward_cfg : weights from config.mdp.reward
    """
    reward = 0.0

    # --- Terminal reward ---
    if is_terminal and outcome is not None:
        if outcome == "survived":
            reward += reward_cfg.get("survival_bonus", 100)
        else:
            reward += reward_cfg.get("death_penalty", -100)

    # --- Oxygenation improvement ---
    spo2_delta = state_t1.get("spo2", 0) - state_t.get("spo2", 0)
    reward += reward_cfg.get("spo2_weight", 2.0) * spo2_delta

    # PF ratio improvement
    if "pf_ratio" in state_t1 and "pf_ratio" in state_t:
        pf_delta = state_t1["pf_ratio"] - state_t["pf_ratio"]
        reward += reward_cfg.get("pf_ratio_weight", 0.1) * pf_delta

    # MAP improvement
    map_delta = state_t1.get("mean_arterial_pressure", 0) - state_t.get("mean_arterial_pressure", 0)
    if map_delta > 0:
        reward += reward_cfg.get("map_improvement_weight", 0.5) * map_delta

    # --- Safety penalties ---
    pp = state_t1.get("plateau_pressure", 0)
    if pp > 30:
        reward += reward_cfg.get("plateau_pressure_30_penalty", -15)
    elif pp > 28:
        reward += reward_cfg.get("plateau_pressure_28_penalty", -5)

    dp = state_t1.get("driving_pressure", 0)
    if dp > 15:
        reward += reward_cfg.get("driving_pressure_15_penalty", -8)

    mp = state_t1.get("mechanical_power", 0)
    if mp > 20:
        penalty_per = reward_cfg.get("mp_over_20_penalty_per_unit", -3)
        reward += penalty_per * (mp - 20)

    spo2 = state_t1.get("spo2", 100)
    if spo2 < 85:
        reward += reward_cfg.get("spo2_below_85_penalty", -25)
    elif spo2 < 88:
        reward += reward_cfg.get("spo2_below_88_penalty", -10)

    map_val = state_t1.get("mean_arterial_pressure", 80)
    if map_val < 60:
        reward += reward_cfg.get("map_below_60_penalty", -20)

    paco2 = state_t1.get("paco2", 40)
    if paco2 > 60:
        reward += reward_cfg.get("paco2_over_60_penalty", -10)

    # --- Time penalty ---
    hours_on_vent = state_t.get("hours_on_ventilator", 0)
    reward += reward_cfg.get("time_penalty_per_hour", -0.1) * hours_on_vent

    return reward


# ===================================================================
# Episode construction
# ===================================================================
def build_episodes(
    df: pd.DataFrame,
    config: dict,
    feature_cols: list[str],
) -> list[dict]:
    """
    Convert a preprocessed hourly DataFrame into a list of episode dicts.

    Each episode:
        {"stay_id": int, "outcome": str,
         "transitions": [
            {"state": np.ndarray, "action": int, "reward": float,
             "next_state": np.ndarray, "terminal": bool}, ...
         ]}
    """
    mdp_cfg = config["mdp"]
    action_bins = mdp_cfg["actions"]
    reward_cfg = mdp_cfg["reward"]
    episodes = []

    for stay_id, group in df.groupby("stay_id"):
        group = group.sort_values("hour_index").reset_index(drop=True)
        if len(group) < 2:
            continue

        outcome = "survived" if group.iloc[0].get("hospital_expire_flag", 0) == 0 else "died"
        transitions = []

        for t in range(len(group) - 1):
            row_t = group.iloc[t]
            row_t1 = group.iloc[t + 1]

            state = row_t[feature_cols].values.astype(np.float32)
            next_state = row_t1[feature_cols].values.astype(np.float32)

            # Compute observed action (MP delta)
            mp_delta = row_t1.get("mechanical_power", 0) - row_t.get("mechanical_power", 0)
            action = discretise_action(mp_delta, action_bins)

            is_terminal = t + 1 == len(group) - 1
            reward = calculate_reward(
                state_t=row_t.to_dict(),
                state_t1=row_t1.to_dict(),
                is_terminal=is_terminal,
                outcome=outcome if is_terminal else None,
                reward_cfg=reward_cfg,
            )

            transitions.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminal": is_terminal,
                }
            )

        episodes.append({"stay_id": stay_id, "outcome": outcome, "transitions": transitions})

    logger.info(f"Built {len(episodes)} episodes with {sum(len(e['transitions']) for e in episodes)} transitions")
    return episodes


# ===================================================================
# Train / val / test split
# ===================================================================
def split_episodes(
    episodes: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Patient-level stratified split into train / val / test.
    """
    split_cfg = config["training"]["split"]
    train_ratio = split_cfg["train"]
    val_ratio = split_cfg["val"]

    outcomes = np.array([1 if e["outcome"] == "died" else 0 for e in episodes])
    stay_ids = np.array([e["stay_id"] for e in episodes])
    indices = np.arange(len(episodes))

    # First split: train vs (val+test)
    from sklearn.model_selection import train_test_split

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=outcomes,
        random_state=42,
    )

    # Second split: val vs test
    temp_outcomes = outcomes[temp_idx]
    relative_val = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - relative_val,
        stratify=temp_outcomes,
        random_state=42,
    )

    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    test_eps = [episodes[i] for i in test_idx]

    logger.info(
        f"Split: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}"
    )
    return train_eps, val_eps, test_eps


# ===================================================================
# Convert episodes → flat arrays for d3rlpy / sklearn
# ===================================================================
def episodes_to_arrays(
    episodes: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten episodes into arrays suitable for d3rlpy MDPDataset.

    Returns (observations, actions, rewards, next_observations, terminals)
    """
    obs, acts, rews, next_obs, terms = [], [], [], [], []

    for ep in episodes:
        for tr in ep["transitions"]:
            obs.append(tr["state"])
            acts.append(tr["action"])
            rews.append(tr["reward"])
            next_obs.append(tr["next_state"])
            terms.append(float(tr["terminal"]))

    return (
        np.array(obs, dtype=np.float32),
        np.array(acts, dtype=np.int64),
        np.array(rews, dtype=np.float32),
        np.array(next_obs, dtype=np.float32),
        np.array(terms, dtype=np.float32),
    )
