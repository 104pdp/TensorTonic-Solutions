import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

    # Convert inputs to NumPy arrays (keeps scalars working)
    param = np.asarray(param, dtype=float)
    grad  = np.asarray(grad, dtype=float)
    m     = np.asarray(m, dtype=float)
    v     = np.asarray(v, dtype=float)

    # 1. Update biased first and second moments
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # 2. Bias correction (t is 1-based)
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)

    # 3. Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new
