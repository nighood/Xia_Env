import gymnasium
from .base import Xia1

gymnasium.envs.register(
    id=f"active-search-v0",
    entry_point=f"activesearch:Xia1",
)
