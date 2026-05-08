"""Sanity check for FetchEnvWrapper + HERReplayBuffer integration."""
import sys
sys.path.append('.')
from environments.fetch_wrapper import FetchEnvWrapper
from agents.her_buffer import HERReplayBuffer


def main():
    # ---- Test 1: wrapper instantiation ----
    env = FetchEnvWrapper("FetchPush-v4")
    print(f"Test 1: dimensions")
    print(f"  obs_dim={env.obs_dim}, goal_dim={env.goal_dim}, "
          f"action_dim={env.action_dim}, max_steps={env.max_episode_steps}")
    print("  PASS")

    # ---- Test 2: random rollout shape ----
    episode = env.collect_episode()
    T = episode['actions'].shape[0]
    assert episode['obs'].shape           == (T + 1, env.obs_dim)
    assert episode['achieved_goal'].shape == (T + 1, env.goal_dim)
    assert episode['desired_goal'].shape  == (T,     env.goal_dim)
    assert episode['actions'].shape       == (T,     env.action_dim)
    print(f"Test 2: rollout shape (T={T}): PASS")

    # ---- Test 3: end-to-end flow into buffer ----
    buffer = HERReplayBuffer(
        max_episodes=10,
        max_episode_steps=env.max_episode_steps,
        obs_dim=env.obs_dim,
        goal_dim=env.goal_dim,
        action_dim=env.action_dim,
        compute_reward_fn=env.compute_reward,
        her_k=4,
        device="cpu",
    )

    for _ in range(5):
        buffer.store_episode(env.collect_episode())

    batch = buffer.sample(64)
    print(f"Test 3: end-to-end (5 episodes stored, batch of 64 sampled)")
    print(f"  rewards mean: {batch['rewards'].mean().item():.3f}")
    print(f"  rewards min/max: {batch['rewards'].min().item():.1f}/{batch['rewards'].max().item():.1f}")
    assert batch['obs'].shape          == (64, env.obs_dim)
    assert batch['desired_goal'].shape == (64, env.goal_dim)
    assert batch['rewards'].shape      == (64,)
    print("  PASS")

    print("\nAll env wrapper tests passed.")
    env.close()


if __name__ == "__main__":
    main()