import json
import matplotlib.pyplot as plt

with open('results/metrics.json') as f:
    metrics = json.load(f)

epochs  = [m['epoch'] for m in metrics]
success = [m['success_rate'] * 100 for m in metrics]
q1_mean = [m['q1_mean'] for m in metrics]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(epochs, success, 'b-o', markersize=4)
ax1.axhline(y=70, color='r', linestyle='--', label='70% threshold')
ax1.axvline(x=5, color='g', linestyle='--', alpha=0.7, label='Warmup ends')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Success Rate (%)')
ax1.set_title('HER+SAC on FetchPush-v4: Success Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 105)

ax2.plot(epochs, q1_mean, 'r-o', markersize=4)
ax2.axvline(x=5, color='g', linestyle='--', alpha=0.7, label='Warmup ends')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean Q-Value')
ax2.set_title('Q-Value Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curve.png', dpi=150, bbox_inches='tight')
print("Saved to results/training_curve.png")