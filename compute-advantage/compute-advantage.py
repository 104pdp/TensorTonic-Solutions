import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    T = len(rewards)
    rewards = np.asarray(rewards, dtype=float)
    V = np.asarray(V, dtype=float)

    advantages = np.zeros(T, dtype=float)

    G = 0.0
    # Compute returns backward
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        advantages[t] = G - V[states[t]]

    return advantages
