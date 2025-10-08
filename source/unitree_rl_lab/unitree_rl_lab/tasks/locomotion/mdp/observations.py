from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def distance_hand_object(env, env_ids=None, asset_cfg=None, palm_link_name="right_hand_base_link"):
    """Euklidovská vzdálenost mezi dlaní (base link ruky) a kostkou."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # najdeme index base linku ruky
    try:
        link_id = env.scene["robot"].data.body_names.index(palm_link_name)
    except ValueError:
        # fallback – pokud se název změní, vezmeme první prst jako approx
        print(f"[WARN] palm_link_name '{palm_link_name}' not found, falling back to R_index_proximal")
        link_id = env.scene["robot"].data.body_names.index("R_index_proximal")

    palm_pos = env.scene["robot"].data.body_pos_w[env_ids, link_id, :]  # (N,3)
    obj_pos = env.scene[asset_cfg.name].data.body_pos_w[env_ids, 0, :]  # (N,3)

    return torch.norm(palm_pos - obj_pos, dim=-1)



def distance_object_target(env, env_ids=None, object_cfg=None, target_cfg=None):
    """Euklidovská vzdálenost mezi kostkou a target square."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    obj_pos = env.scene[object_cfg.name].data.body_pos_w[env_ids, 0, :]
    tgt_pos = env.scene[target_cfg.name].data.body_pos_w[env_ids, 0, :]

    return torch.norm(obj_pos - tgt_pos, dim=-1)  # (N,)

def is_grasped(env, env_ids=None, object_cfg=None, table_z=0.84, lift_eps=0.02):
    """
    Heuristika: kostka je nad stolem a nepohybuje se moc rychle.
    (Později můžeš rozšířit o check kontaktů prstů.)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    obj_pos = env.scene[object_cfg.name].data.body_pos_w[env_ids, 0, :]
    obj_vel = env.scene[object_cfg.name].data.body_vel_w[env_ids, 0, :]

    lifted = obj_pos[:, 2] > (table_z + lift_eps)
    still = torch.norm(obj_vel, dim=-1) < 0.2

    return (lifted & still).bool()

def is_placed(env, env_ids=None, object_cfg=None, target_cfg=None,
              xy_tol=0.03, z_tol=0.03, vel_tol=0.05):
    """Úspěch: kostka je v target square (XY+Z tolerance) a stojí."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    obj_pos = env.scene[object_cfg.name].data.body_pos_w[env_ids, 0, :]
    obj_vel = env.scene[object_cfg.name].data.body_vel_w[env_ids, 0, :]
    tgt_pos = env.scene[target_cfg.name].data.body_pos_w[env_ids, 0, :]

    xy_ok = torch.norm(obj_pos[:, :2] - tgt_pos[:, :2], dim=-1) < xy_tol
    z_ok = torch.abs(obj_pos[:, 2] - tgt_pos[:, 2]) < z_tol
    still = torch.norm(obj_vel, dim=-1) < vel_tol

    return (xy_ok & z_ok & still).bool()


def object_root_pos(env, env_ids=None, asset_cfg=None):
    """Return (x,y,z) pos of object."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    # vezmeme první rigid body objektu (kostka má 1 link = 0)
    return env.scene[asset_cfg.name].data.body_pos_w[env_ids, 0, :]

def object_root_vel(env, env_ids=None, asset_cfg=None):
    """Return linear velocity (vx,vy,vz) of object."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    return env.scene[asset_cfg.name].data.body_vel_w[env_ids, 0, :]

def target_root_pos(env, env_ids=None, asset_cfg=None):
    """Return (x,y,z) pos of target square (also rigid object/asset)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    return env.scene[asset_cfg.name].data.body_pos_w[env_ids, 0, :]

def object_below_height(env, asset_cfg, min_height=0.6):
    obj_pos = env.scene[asset_cfg.name].data.root_pos_w
    return obj_pos[:, 2] < min_height