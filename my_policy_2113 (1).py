import numpy as np

def policy_action(params, observation):
    # Linear mapping: 8 inputs x 4 actions = 32 weights + 4 biases = 36 parameters
    W = params[:32].reshape(8, 4)
    b = params[32:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)