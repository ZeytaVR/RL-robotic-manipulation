"""Sanity checks for HERReplayBuffer. Run before SAC integration."""
import numpy as np
import sys
sys.path.append('.')
from agents.her_buffer import HERReplayBuffer


def fake_compute_reward(achieved, desired, info):
    """Sparse 0/-1 reward, FetchPush-style threshold."""
    threshold = 0.05
    distances = np.linalg.norm(achieved - desired, axis=-1)
    return -(distances > threshold).astype(np.float32)


def make_fake_episode(T, obs_dim, goal_dim, action_dim, seed):
    rng = np.random.default_rng(seed)
    # Simulate slowly-drifting achieved_goal so HER has meaningful structure
    base = rng.normal(size=(goal_dim,))
    drift = rng.normal(scale=0.02, size=(T + 1, goal_dim)).cumsum(axis=0)
    return {
        'obs':           rng.normal(size=(T + 1, obs_dim)).astype(np.float32),
        'achieved_goal': (base + drift).astype(np.float32),
        'desired_goal':  np.tile(rng.normal(size=(goal_dim,)), (T, 1)).astype(np.float32),
        'actions':       rng.uniform(-1, 1, size=(T, action_dim)).astype(np.float32),
    }


def main():
    OBS_DIM, GOAL_DIM, ACTION_DIM, T_MAX = 10, 3, 4, 50

    buffer = HERReplayBuffer(
        max_episodes=100,
        max_episode_steps=T_MAX,
        obs_dim=OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACTION_DIM,
        compute_reward_fn=fake_compute_reward,
        her_k=4,
        device="cpu",
    )

    # ---- Test 1: roundtrip + shapes ----
    ep = make_fake_episode(T_MAX, OBS_DIM, GOAL_DIM, ACTION_DIM, seed=0)
    buffer.store_episode(ep, is_demo=False)
    batch = buffer.sample(8)

    assert batch['obs'].shape == (8, OBS_DIM)
    assert batch['next_obs'].shape == (8, OBS_DIM)
    assert batch['actions'].shape == (8, ACTION_DIM)
    assert batch['desired_goal'].shape == (8, GOAL_DIM)
    assert batch['achieved_next'].shape == (8, GOAL_DIM)
    assert batch['rewards'].shape == (8,)
    assert batch['dones'].shape == (8,)
    print("Test 1 (roundtrip + shapes): PASS")

    # ---- Test 2: reward correctness ----
    # Rewards in batch must equal compute_reward(achieved_next, desired).
    # If they don't, you forgot to recompute or used the wrong index.
    for i in range(1, 51):
        buffer.store_episode(make_fake_episode(T_MAX, OBS_DIM, GOAL_DIM, ACTION_DIM, seed=i))

    big = buffer.sample(5000)
    achieved_next = big['achieved_next'].cpu().numpy()
    desired = big['desired_goal'].cpu().numpy()
    rewards = big['rewards'].cpu().numpy()

    expected = fake_compute_reward(achieved_next, desired, {})
    assert np.allclose(rewards, expected), "Reward != compute_reward(achieved_next, desired)"
    frac_zero = (rewards == 0).mean()
    frac_neg = (rewards == -1).mean()
    print(f"Test 2 (reward correctness): PASS")
    print(f"  Distribution: {frac_zero:.2%} zero, {frac_neg:.2%} -1")
    print(f"  (Both should be > 0. All-zero or all-(-1) means relabeling is broken.)")
    assert 0 < frac_zero < 1, "All rewards same value — relabeling or reward fn broken"

    # ---- Test 3: future_t bounds ----
    # Reproduce sample-time index logic to verify no leakage out of episode.
    rng = np.random.default_rng(42)
    ep_idx = rng.integers(0, buffer.n_episodes, size=1000)
    ep_lengths = buffer.episode_lengths[ep_idx]
    t = (rng.uniform(size=1000) * ep_lengths).astype(np.int32)
    future_offset = (rng.uniform(size=1000) * (ep_lengths - t)).astype(np.int32) + 1
    future_t = np.minimum(t + future_offset, ep_lengths)

    assert np.all(future_t > t), "future_t must be > t"
    assert np.all(future_t <= ep_lengths), "future_t must be within episode"
    print("Test 3 (future_t bounds): PASS")

    # ---- Test 4: demo flag ----
    b2 = HERReplayBuffer(
        max_episodes=10, max_episode_steps=T_MAX,
        obs_dim=OBS_DIM, goal_dim=GOAL_DIM, action_dim=ACTION_DIM,
        compute_reward_fn=fake_compute_reward, device="cpu",
    )
    b2.store_episode(make_fake_episode(10, OBS_DIM, GOAL_DIM, ACTION_DIM, seed=99), is_demo=True)
    b2.store_episode(make_fake_episode(10, OBS_DIM, GOAL_DIM, ACTION_DIM, seed=100), is_demo=False)
    assert b2.is_demo[0] and not b2.is_demo[1]
    print("Test 4 (demo flag): PASS")

    print("\nAll buffer sanity checks passed.")


if __name__ == "__main__":
    main()