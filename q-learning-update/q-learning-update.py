import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.asarray(Q, dtype=float)
    Q_new = Q.copy()

    target = float(r) + float(gamma) * float(np.max(Q[s_next]))
    Q_new[s, a] = Q[s, a] + float(alpha) * (target - Q[s, a])

    return Q_new
