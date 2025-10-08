# unitree_rl_lab/tasks/locomotion/robots/g1/29dof/pickplace_redblock_rl_env.py

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .pickplace_redblock_rl_env_cfg import PickPlaceRedBlockRLEnvCfg


class PickPlaceRedBlockRLEnv(ManagerBasedRLEnv):
    """RL env s barevnými markery nad každým robotem podle aktuální odměny."""
    cfg_cls = PickPlaceRedBlockRLEnvCfg

    def __init__(self, cfg: PickPlaceRedBlockRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)


    def step(self, action):
        # --- zavolej standardní krok ---
        obs, rew, terminated, truncated, info = super().step(action)

        # --- každých X kroků vypiš podrobné rewardy pro všechny envy ---
        if self.common_step_counter % 200 == 0:  # interval ladění
            num_envs = self.num_envs
            print(f"\n[Step {self.common_step_counter}] Reward breakdown:")
            for env_i in range(num_envs):
                terms = self.reward_manager.get_active_iterable_terms(env_i)
                total = sum(val[0] for _, val in terms)
                print(f"  Env {env_i:02d} | total = {total:+.3f}")
                for name, val in terms:
                    print(f"     {name:<20}: {val[0]:+.4f}")
            print("------------------------------------------------------")

        return obs, rew, terminated, truncated, info
