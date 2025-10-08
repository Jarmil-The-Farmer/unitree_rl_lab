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

        # ---- Marker prototypy: pořadí určuje indexy (neg=0, pos=1, zero=2) ----
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/RewardMarkers",
            markers={
                "neg": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                ),
                "pos": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                ),
                "zero": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
                ),
            },
        )
        self._reward_markers = VisualizationMarkers(marker_cfg)

        # předpočítáme index „pelvis“ linku (pro pozici nad robotem)
        self._pelvis_id = self.scene["robot"].data.body_names.index("pelvis")
        self._offset = torch.tensor([0.0, 0.0, 1.5], device=self.device)

        # pro mapování indexů markerů (neg=0, pos=1, zero=2)
        self._idx_neg = torch.tensor(0, device=self.device, dtype=torch.long)
        self._idx_pos = torch.tensor(1, device=self.device, dtype=torch.long)
        self._idx_zero = torch.tensor(2, device=self.device, dtype=torch.long)

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)

        # Pozice markerů: nad „pelvis“ každého roba
        pelvis_pos = self.scene["robot"].data.body_pos_w[:, self._pelvis_id, :]  # (N,3)
        translations = pelvis_pos + self._offset  # (N,3)

        # Vyber prototyp podle znamenka odměny (indexy dle pořadí v markers dict)
        idx = torch.where(
            rew > 1e-5, self._idx_pos, torch.where(rew < -1e-5, self._idx_neg, self._idx_zero)
        )  # (N,)

        # vykresli (orientace a scale nejsou nutné, pokud je nechceš měnit)
        self._reward_markers.visualize(translations=translations, marker_indices=idx)

        return obs, rew, terminated, truncated, info
