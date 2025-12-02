

import numpy as np
from surface_code_env import SurfaceCodeEnv  

# Initialize the environment
env = SurfaceCodeEnv(
    d=5,
    p_phys=0.2,
    error_model='X',
    include_masks=False,
    max_n_steps=100
)

# Reset the environment
state, _ = env.reset(seed=42)
done = False

print("Random agent test started!\n")

while not done:
    # Sample a random action from the action space
    action = env.action_space.sample()
    
    # Step the environment
    next_state, reward, done, info = env.step(action)
    
    # Render the current state
    env.render(figsize=6)
    
    print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
    

print("\nRandom agent test finished!")
