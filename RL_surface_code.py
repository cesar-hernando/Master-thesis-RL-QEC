

import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from utils import RewardTrackerCallback, plot_learning, record_video, show_videos
from surface_code_env import SurfaceCodeEnv  


def evaluate(rl_algorithm, eval_env, n_episodes=1, model=None):

    print("\nRandom agent evaluation started!")
    cumulative_rewards = []
    for i in range(n_episodes):
        print(f"\n Episode {i} started")
        # Reset the environment
        state, _ = env.reset()
        done = False
        cumulative_reward = 0
        while not done:
            if rl_algorithm == 'random':
                # Sample a random action from the action space
                action = eval_env.action_space.sample()
            elif rl_algorithm == 'DQN':
                action, _ = model.predict(observation=state)
            else:
                raise(ValueError, f"{rl_algorithm} has not been implemented yet.")
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            done = terminated or truncated
            cumulative_reward += reward
            # Render the current state
            eval_env.render(wait_time=0.0001)
            
            print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
        
        print(f"Cumulative reward = {cumulative_reward}")
        cumulative_rewards.append(cumulative_reward)

    print("\nEvaluation finished!")
    mean_cum_reward = np.mean(cumulative_rewards)
    std_cum_reward = np.std(cumulative_rewards)
    print(f"Mean cumulative reward = {mean_cum_reward} ± {std_cum_reward}\n" )


if __name__ == '__main__':

    rl_algorithm = 'DQN' # 'random' or 'DQN' or 'PPO'
    training_steps = 10_000
    mode = "test"   # change to "test" after training
    n_test_episodes = 20

    # Initialize the environment with preferred settings
    env = SurfaceCodeEnv(
        d=5,
        p_phys=0.1,
        error_model='X',
        include_masks=False,
        max_n_steps=1000
    )

    env = Monitor(env)
    callback = RewardTrackerCallback()

    # Define the learning agent implemented in Stable Baselines using custom hyperparameters
    model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=10_000,
            batch_size=32,
            learning_starts=500,
            train_freq=1,
            gradient_steps=1,
            gamma=0.97,
            tau=1.0,
            target_update_interval=1000,
            exploration_fraction=0.25,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log="./logs/"
            )


    if mode == "train":
        # Training with default hyperparameters
        model.learn(total_timesteps=training_steps, callback=callback, progress_bar=True)
        mean_rewards = callback.mean_rewards
        plot_learning(mean_rewards)
        model.save("./dqn_surface_code")

    elif mode == "test":
        # Load trained model instead of creating a new one
        model = DQN.load("./dqn_surface_code")

        eval_env = SurfaceCodeEnv(
            d=5,
            p_phys=0.1,
            error_model='depolarizing',
            include_masks=False,
            max_n_steps=1000
            )

        evaluate(rl_algorithm=rl_algorithm, eval_env=eval_env, n_episodes=n_test_episodes, model=model)
        #mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        #print(f"\n Mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        #record_video("CartPole-v1", model, video_length=5000, prefix="dqn-surface_code")
        #show_videos("videos", prefix="dqn")

        



