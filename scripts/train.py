"""SAC + HER training loop on FetchPush-v4."""
import sys, os, time, json
sys.path.append('.')
import numpy as np
import torch

from environments.fetch_wrapper import FetchEnvWrapper
from agents.her_buffer import HERReplayBuffer
from agents.sac import SACAgent


# ----- Hyperparameters -----
ENV_ID = "FetchPush-v4"
SEED = 42

N_EPOCHS = 50
N_CYCLES_PER_EPOCH = 50
N_ROLLOUTS_PER_CYCLE = 16
N_UPDATES_PER_CYCLE = 40
N_EVAL_EPISODES = 10
BATCH_SIZE = 256

BUFFER_MAX_EPISODES = 10_000
HER_K = 4

GAMMA = 0.98
TAU = 0.05
HIDDEN_DIM = 256
LR = 3e-4

RESULTS_DIR = "results"
MODEL_DIR = "models"


def save_checkpoint(agent, path, extra=None):
    state = {
        'actor':                agent.actor.state_dict(),
        'q1':                   agent.q1.state_dict(),
        'q2':                   agent.q2.state_dict(),
        'log_alpha':            agent.log_alpha.detach().clone(),
        'obs_normalizer_mean':  agent.obs_normalizer.mean.cpu(),
        'obs_normalizer_std':   agent.obs_normalizer.std.cpu(),
        'goal_normalizer_mean': agent.goal_normalizer.mean.cpu(),
        'goal_normalizer_std':  agent.goal_normalizer.std.cpu(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def evaluate(agent, eval_env, n_episodes):
    successes = 0
    for _ in range(n_episodes):
        obs_dict, _ = eval_env.reset()
        for _ in range(eval_env.max_episode_steps):
            action = agent.select_action(
                obs_dict['observation'], obs_dict['desired_goal'], deterministic=True
            )
            obs_dict, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                break
        if np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal']) < 0.05:
            successes += 1
    return successes / n_episodes


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = FetchEnvWrapper(ENV_ID, seed=SEED)
    eval_env = FetchEnvWrapper(ENV_ID, seed=SEED + 1)

    buffer = HERReplayBuffer(
        max_episodes=BUFFER_MAX_EPISODES,
        max_episode_steps=env.max_episode_steps,
        obs_dim=env.obs_dim, goal_dim=env.goal_dim, action_dim=env.action_dim,
        compute_reward_fn=env.compute_reward,
        her_k=HER_K, device=device,
    )

    agent = SACAgent(
        obs_dim=env.obs_dim, goal_dim=env.goal_dim, action_dim=env.action_dim,
        action_low=env.action_low, action_high=env.action_high,
        hidden_dim=HIDDEN_DIM, gamma=GAMMA, tau=TAU,
        actor_lr=LR, critic_lr=LR, alpha_lr=LR, device=device,
    )

    print(f"Env: {ENV_ID}, dims: obs={env.obs_dim}, goal={env.goal_dim}, action={env.action_dim}")
    print(f"Planned env steps: {N_EPOCHS * N_CYCLES_PER_EPOCH * N_ROLLOUTS_PER_CYCLE * env.max_episode_steps:,}")
    print(f"Planned grad updates: {N_EPOCHS * N_CYCLES_PER_EPOCH * N_UPDATES_PER_CYCLE:,}")
    print()

    metrics = []
    best_success = 0.0
    start = time.time()

    def explore_policy(obs, goal):
        action = agent.select_action(obs, goal, deterministic=False)
        if np.random.uniform() < 0.3:  # random_eps
            action = np.random.uniform(env.action_low, env.action_high, size=env.action_dim)
        else:
            noise_scale = 0.2 * (env.action_high - env.action_low)  # noise_eps
            action = action + noise_scale * np.random.normal(size=env.action_dim)
        return np.clip(action, env.action_low, env.action_high)

    def warmup_policy(obs, goal):
        """Steer gripper toward puck, then push with noise. Ensures contact."""
        rel_pos = obs[6:9]  # puck_pos - gripper_pos
        action = np.zeros(4)
        action[:3] = rel_pos * 5.0  # move toward puck
        action[:3] += np.random.normal(size=3) * 0.5  # directional noise
        action[3] = np.random.uniform(-1, 1)  # random gripper
        return np.clip(action, -1, 1)

    for epoch in range(N_EPOCHS):
        ep_losses = {k: [] for k in ['q1_loss', 'q2_loss', 'actor_loss', 'alpha_loss', 'alpha', 'q1_mean']}

        for _ in range(N_CYCLES_PER_EPOCH):
            for _ in range(N_ROLLOUTS_PER_CYCLE):
                if epoch < 5:
                    episode = env.collect_episode(policy_fn=warmup_policy)
                else:
                    episode = env.collect_episode(policy_fn=explore_policy)
                buffer.store_episode(episode)
                agent.update_normalizers(episode)

            for _ in range(N_UPDATES_PER_CYCLE):
                batch = buffer.sample(BATCH_SIZE)
                losses = agent.update(batch)
                for k in ep_losses:
                    if k in losses:
                        ep_losses[k].append(losses[k])

        success_rate = evaluate(agent, eval_env, N_EVAL_EPISODES)
        elapsed = time.time() - start
        avg = {k: float(np.mean(v)) for k, v in ep_losses.items() if v}

        print(
            f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
            f"success={success_rate:.2%} | "
            f"q1_mean={avg.get('q1_mean', 0):.2f} | "
            f"actor_loss={avg.get('actor_loss', 0):.3f} | "
            f"alpha={avg.get('alpha', 0):.3f} | "
            f"elapsed={elapsed/60:.1f}min"
        )

        metrics.append({'epoch': epoch + 1, 'success_rate': success_rate,
                        'elapsed_seconds': elapsed, **avg})
        with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        save_checkpoint(agent, os.path.join(MODEL_DIR, "latest.pt"),
                        extra={'epoch': epoch + 1, 'success_rate': success_rate})
        if success_rate > best_success:
            best_success = success_rate
            save_checkpoint(agent, os.path.join(MODEL_DIR, "best.pt"),
                            extra={'epoch': epoch + 1, 'success_rate': success_rate})

    print(f"\nTraining complete. Best success rate: {best_success:.2%}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()