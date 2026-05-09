"""Load best.pt and run/record evaluation episodes."""
import sys, os
sys.path.append('.')
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics  # noqa
from agents.networks import Actor
from agents.normalizer import Normalizer
from environments.fetch_wrapper import FetchEnvWrapper

MODEL_PATH  = "models/best.pt"
ENV_ID      = "FetchPush-v4"
N_EPISODES  = 10
VIDEO_PATH  = "results/eval_video.mp4"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(path, env):
    ckpt = torch.load(path, map_location=DEVICE)
    state_dim = env.obs_dim + env.goal_dim
    actor = Actor(state_dim, env.action_dim, 256,
                  env.action_low, env.action_high).to(DEVICE)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    obs_norm  = Normalizer(env.obs_dim,  device=DEVICE)
    goal_norm = Normalizer(env.goal_dim, device=DEVICE)
    obs_norm.mean  = ckpt['obs_normalizer_mean'].to(DEVICE)
    obs_norm.std   = ckpt['obs_normalizer_std'].to(DEVICE)
    goal_norm.mean = ckpt['goal_normalizer_mean'].to(DEVICE)
    goal_norm.std  = ckpt['goal_normalizer_std'].to(DEVICE)
    return actor, obs_norm, goal_norm


def act(obs, goal, actor, obs_norm, goal_norm):
    obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    goal_t = torch.from_numpy(goal).float().unsqueeze(0).to(DEVICE)
    state  = torch.cat([obs_norm.normalize(obs_t),
                        goal_norm.normalize(goal_t)], dim=-1)
    with torch.no_grad():
        return actor.deterministic_action(state).cpu().numpy()[0]


def main():
    ref_env  = FetchEnvWrapper(ENV_ID)
    actor, obs_norm, goal_norm = load_policy(MODEL_PATH, ref_env)
    ref_env.close()

    render_env = gym.make(ENV_ID, render_mode='rgb_array')
    os.makedirs("results", exist_ok=True)
    frames, successes = [], 0

    for ep in range(N_EPISODES):
        obs_dict, _ = render_env.reset()
        for _ in range(render_env.spec.max_episode_steps):
            frames.append(render_env.render())
            action   = act(obs_dict['observation'], obs_dict['desired_goal'],
                           actor, obs_norm, goal_norm)
            obs_dict, _, terminated, truncated, _ = render_env.step(action)
            if terminated or truncated:
                break

        dist    = np.linalg.norm(
            obs_dict['achieved_goal'] - obs_dict['desired_goal'])
        success = dist < 0.05
        successes += success
        print(f"Episode {ep+1:2d}: {'SUCCESS' if success else 'FAIL'} "
              f"(dist={dist:.3f})")

    print(f"\nSuccess rate: {successes}/{N_EPISODES} "
          f"({100*successes/N_EPISODES:.0f}%)")

    try:
        import imageio
        imageio.mimsave(VIDEO_PATH, frames, fps=25)
        print(f"Video saved → {VIDEO_PATH}")
    except ImportError:
        print("imageio not installed. Run: pip install imageio[ffmpeg]")
        print(f"Saving {len(frames)} individual frames instead...")
        for i, f in enumerate(frames):
            import PIL.Image
            PIL.Image.fromarray(f).save(f"results/frame_{i:04d}.png")

    render_env.close()


if __name__ == "__main__":
    main()