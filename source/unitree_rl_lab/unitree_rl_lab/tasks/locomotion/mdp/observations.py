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

        # vypočítej vzdálenost
    distances = torch.norm(palm_pos - obj_pos, dim=-1)  # (N,)

    # --- DEBUG výpis ---
    # každých pár kroků (ne každou ms), aby to nebylo zahlcené
    if getattr(env, "common_step_counter", 0) % 10 == 0:
        dist_cpu = distances.detach().cpu().numpy()
        print("\n[DEBUG] Distance hand ↔ object per env:")
        for i, d in zip(env_ids.tolist(), dist_cpu):
            print(f"  Env {i:02d}: {d:.4f} m")

    return distances



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

def is_grasped_stable(
    env,
    env_ids=None,
    object_cfg=None,
    contact_sensor_name: str = "cube_hand_contacts",
    contact_force_eps: float = 0.8,      # N – práh detekce kontaktu s rukou
    vel_thresh: float = 0.10,            # m/s – kostka se moc nehýbe
    grasp_height_offset: float = 0.02    # m – nad stolem
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # 1) kontakty kostka×ruka (filtrované)
    sensor = env.scene.sensors[contact_sensor_name]
    # force_matrix_w: (num_env, B=1, M, 3) => vezmeme normu přes xyz a any() přes M
    F = sensor.data.force_matrix_w[env_ids, 0]          # (num_env, M, 3)
    has_hand_contact = torch.any(torch.linalg.norm(F, dim=-1) > contact_force_eps, dim=-1)

    # 2) klidnost a výška nad stolem
    obj_pos = env.scene[object_cfg.name].data.body_pos_w[env_ids, 0, :]
    obj_vel = torch.linalg.norm(env.scene[object_cfg.name].data.body_vel_w[env_ids, 0, :], dim=-1)
    if hasattr(env.scene["packing_table"], "data"):
        table_z = env.scene["packing_table"].data.body_pos_w[env_ids, 0, 2]
    else:
        table_z = torch.full((len(env_ids),), 0.84, device=env.device)

    above_table = obj_pos[:, 2] > (table_z + grasp_height_offset)
    still = obj_vel < vel_thresh

    return (has_hand_contact & above_table & still).float()

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



def self_collision_penalty(env, asset_cfg, force_threshold: float = 1.0):
    """
    Penalizace za kolizi robota se sebou samým.
    """
    contact_forces = env.scene[asset_cfg.name].data.net_contact_forces_w
    force_norm = torch.norm(contact_forces, dim=-1)
    collisions = force_norm > force_threshold
    penalty = collisions.float().sum(dim=-1) * -1.0
    return penalty


def has_self_collision(env, asset_cfg, threshold: float = 50.0):
    """
    Ukončí epizodu, pokud robot koliduje sám se sebou s velkou silou.
    """
    contact_forces = env.scene[asset_cfg.name].data.net_contact_forces_w
    force_norm = torch.norm(contact_forces, dim=-1)
    max_force, _ = torch.max(force_norm, dim=-1)
    return max_force > threshold