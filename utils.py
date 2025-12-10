import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import BaseCallback
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import gymnasium as gym


class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []

    def _on_step(self):
        # SB3 logs episode reward in info dict under 'episode' (if env is wrapped)
        info = self.locals["infos"][0]
        
        if "episode" in info:
            ep_reward = info["episode"]["r"]
            self.episode_rewards.append(ep_reward)

            # Compute rolling mean over last N episodes
            window = 20  # choose any window size
            mean_r = np.mean(self.episode_rewards[-window:])
            self.mean_rewards.append(mean_r)

        return True
    

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning(mean_rewards, distance, error_model, smoothing_weight=0.05):
    """
    Plots the learning curve looking like a professional RL paper.
    Following the user's instructions, this plots the raw data with alpha=0.2
    and the smoothed trend line on top with alpha=1.
    
    Parameters
    ----------
    mean_rewards : list or np.array
        The list of mean rewards to plot.
    distance : int
        Code distance (d).
    error_model : str
        Error model name.
    smoothing_weight : float
        The alpha parameter for exponential smoothing.
        Set very low (e.g., 0.05) for strong smoothing.
    """
    
    raw_rewards = np.array(mean_rewards)
    
    # 1. Calculate Heavy Exponential Moving Average (EMA) for the trend line
    smoothed_rewards = []
    last = raw_rewards[0]
    for val in raw_rewards:
        # S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        smoothed_val = smoothing_weight * val + (1 - smoothing_weight) * last
        smoothed_rewards.append(smoothed_val)
        last = smoothed_val
    smoothed_rewards = np.array(smoothed_rewards)
        
    plt.figure(figsize=(10, 6))
    
    # Use a nice color scheme (e.g., a deep blue/purple)
    main_color = '#1f77b4' 

    # 2. Plot the raw data first with low alpha
    # This creates the background "shade" effect from the noisy data itself.
    plt.plot(raw_rewards, color=main_color, alpha=0.2, linewidth=1, label='Raw Data')

    # 3. Plot the smoothed trend line on top with alpha=1 (default)
    # This creates the clear, main trend line.
    plt.plot(smoothed_rewards, color=main_color, linewidth=2.5, alpha=1.0, label='Smoothed Trend (EMA)')

    plt.title(f"Learning Curve (d={distance}, {error_model})", fontsize=14)
    plt.xlabel("Steps (x 1000)", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.legend(loc='best')
    
    # Add a subtle grid
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
    
    # Ensure directory exists
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
        
    # Save high-resolution plot
    save_path = f'./plots/learning_curve_d{distance}_{error_model}_smooth_overlay.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    

def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def record_video(eval_env, model, video_length=500, prefix="", video_folder="./"):
    """
    :param eval_env: (obj)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: eval_env])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()
    
