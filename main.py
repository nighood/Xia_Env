import gymnasium
import registry
env = gymnasium.make("xia-env-v0")
obs, info = env.reset()
# array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

for _ in range(200):
    dummy_action = env.action_space.sample()    # Box(0.0, 1.0, (4,), float32)
    
    observation, rewards, terminated, truncated, info = env.step(dummy_action)
    print("rewards = ", rewards)
    env.render()
