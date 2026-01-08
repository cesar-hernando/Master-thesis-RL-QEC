

import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from utils import SB3Env, RewardTrackerCallback, plot_learning, record_video, show_videos
from surface_code_env import SurfaceCodeEnv  
from neural_network import CNN


import numpy as np

def evaluate_gym_env(model, d=3, error_model="X", include_masks=False, max_n_steps=100, 
                     n_episodes=10, wait_time=0.2, render=False, verbose=False):
    
    env = SurfaceCodeEnv(d=d, p_phys=0.1, error_model=error_model, 
                         include_masks=include_masks, max_n_steps=max_n_steps)
    print("\nAgent evaluation started!")
    success_rate = 0
    ep_scores = []
    n_steps_list = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_score = 0.0
        n_steps = 0

        while not done:
            if render:
                env.render(wait_time=wait_time)  # will show success banner correctly

            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
            action, _ = model.predict(obs_chw, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            ep_score += reward
            n_steps += 1

        if env.status == 1:
            success_rate += 1

        if render:
            # final render to show terminal message
            env.render(wait_time=wait_time)

        if verbose:
            print(f"Ep {ep}: Score={ep_score:.3f}, steps={n_steps}, status={env.status}")

        ep_scores.append(ep_score)
        n_steps_list.append(n_steps)

    print("\nEvaluation finished!")

    mean_ep_score = np.mean(ep_scores)
    std_ep_score = np.std(ep_scores)
    print(f"Mean episode score = {mean_ep_score} ± {std_ep_score}" )

    mean_n_steps = np.mean(n_steps_list)
    std_n_steps = np.std(n_steps_list)
    print(f"Mean number of steps = {mean_n_steps} ± {std_n_steps}" )
    
    print(f"Success rate: {success_rate*100/n_episodes}%\n")



if __name__ == '__main__':
    distance = 5
    error_model = 'depolarizing'  # 'X' or 'Z' or 'depolarizing'
    p_phys = 0.1
    include_masks = True
    max_n_steps = 100
    rl_algorithm = 'DQN' # 'random' or 'DQN' or 'PPO'
    training_steps = 3_000_000
    mode = "train"   # change to "test" after training
    n_test_episodes = 20 # 10_000 for average score, 20 for render demo
    render = False  # whether to render the environment during evaluation
    wait_time = 1.0  # time between steps when rendering
    verbose = False # whether to print detailed info during evaluation

    # policy_kwargs = dict(net_arch=[128, 128]) # for MLP
    policy_kwargs = dict(
        normalize_images=False,      
        features_extractor_class=CNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128],                    # FC head after CNN features (Q-network head)
    )

    
       

    if mode == "train":
        # 1. Initialize the environment with preferred settings, and make it compatible with SB3
        env_kwargs = dict(
            d=distance,
            p_phys=p_phys,
            error_model=error_model,
            include_masks=include_masks,
            max_n_steps=max_n_steps,
        )
        env = SB3Env(SurfaceCodeEnv, env_kwargs, n_envs=1, seed=0, transpose_for_cnn=True)
        #env = SB3Env(SurfaceCodeEnv, env_kwargs, transpose_for_cnn=False) # for MLP

        # 2. Define the learning agent implemented in Stable Baselines using custom hyperparameters
        model = DQN(
                policy="CnnPolicy",
                env=env,
                learning_rate=1e-4,
                buffer_size=100_000,
                batch_size=64,
                learning_starts=10_000,
                train_freq=4,
                gradient_steps=1,
                gamma=0.99,
                tau=1.0,
                target_update_interval=2_000,
                exploration_fraction=0.5,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./logs/"
                )
        
        # 3. Define the reward tracker callback to log rewards during training
        callback = RewardTrackerCallback()
        
        # 4. Training with default hyperparameters
        model.learn(total_timesteps=training_steps, callback=callback, progress_bar=True)

        # 5. Plot learning curve
        mean_rewards = callback.mean_rewards
        plot_learning(mean_rewards, distance, error_model)

        # 6. Save the trained model
        model.save(f"./dqn_surface_code_d{distance}_{error_model}_penalize_rep")

    elif mode == "test":
        # 1. Load traiededned model instead of creating a new one
        model = DQN.load(f"./dqn_surface_code_d{distance}_{error_model}") 

        # 2. Evaluate the agent
        evaluate_gym_env(model, d=distance, error_model=error_model, include_masks=include_masks, max_n_steps=max_n_steps,
                         n_episodes=n_test_episodes, wait_time=wait_time, render=render, verbose=verbose)


        #record_video("CartPole-v1", model, video_length=5000, prefix="dqn-surface_code")
        #show_videos("videos", prefix="dqn")

        



