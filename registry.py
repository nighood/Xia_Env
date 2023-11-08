import gymnasium

gymnasium.envs.register(
    id=f"xia-env-v0",
    entry_point=f"base:Xia1",
)
