import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-PickPlace-RedBlock",
    entry_point="unitree_rl_lab.tasks.locomotion.robots.g1.29dof.pickplace_redblock_rl_env:PickPlaceRedBlockRLEnv",
    disable_env_checker=True,
    kwargs={
        # odkaz na náš RL config
        "env_cfg_entry_point": f"{__name__}.pickplace_redblock_rl_env_cfg:PickPlaceRedBlockRLEnvCfg",
        # play_env_cfg_entry_point – pokud nemáš odlehčenou verzi, můžeš vynechat nebo dát stejnou
        "play_env_cfg_entry_point": f"{__name__}.pickplace_redblock_rl_env_cfg:PickPlaceRedBlockRLEnvPlayCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)