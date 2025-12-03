

import numpy as np
from surface_code_env import SurfaceCodeEnv  


def evaluate(rl_algorithm, eval_env, n_episodes=1):

    print("\nRandom agent evaluation started!")
    mean_cum_reward = 0
    for i in range(n_episodes):
        print(f"\n Episode {i} started")
        # Reset the environment
        state, _ = env.reset(seed=42)
        done = False

        while not done:
            if rl_algorithm == 'random':
                # Sample a random action from the action space
                action = eval_env.action_space.sample()
            else:
                raise(ValueError, f"{rl_algorithm} has not been implemented yet.")
            
            # Step the environment
            next_state, reward, done, info = eval_env.step(action)
            
            # Render the current state
            eval_env.render(wait_time=0.2)
            
            print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
        
        print(f"Cumulative reward = {eval_env.cumulative_reward}")
        mean_cum_reward += eval_env.cumulative_reward

    print("\nEvaluation finished!")
    mean_cum_reward /= n_episodes
    print(f"Mean cumulative reward = {mean_cum_reward} \n" )


if __name__ == '__main__':

    rl_algorithm = 'random'
    n_episodes = 3

    # Initialize the environment with preferred settings
    env = SurfaceCodeEnv(
        d=5,
        p_phys=0.1,
        error_model='depolarizing',
        include_masks=False,
        max_n_steps=100
    )

    # Evaluate the policy learnt
    evaluate(rl_algorithm, env, n_episodes)



