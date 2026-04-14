import gymnasium as gym
import numpy as np
import os

def policy_action(params, observation):
    W = params[:32].reshape(8, 4)
    b = params[32:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

def evaluate_policy(params, episodes=20): 
    total_reward = 0.0
    env = gym.make('LunarLander-v3')
    for _ in range(episodes):
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward
    env.close()
    return total_reward / episodes

def simulated_binary_crossover(p1, p2, eta_c=10):
    child = np.empty(p1.shape[0])
    for i in range(p1.shape[0]):
        u = np.random.rand()
        beta = (2*u)**(1/(eta_c+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta_c+1))
        child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
    return child

def polynomial_mutation(child, mutation_rate=0.25, eta_m=15):
    for i in range(child.shape[0]):
        if np.random.rand() < mutation_rate:
            r = np.random.rand()
            delta_q = (2*r)**(1/(eta_m+1)) - 1 if r < 0.5 else 1 - (2*(1-r))**(1/(eta_m+1))
            child[i] += delta_q * 4.0 
    return np.clip(child, -10, 10)

def train():
    population_size = 150 
    num_generations = 300 
    gene_size = 36        
    filename = "best_policy_2113.npy"
    
    # --- THIS IS THE PART YOU ASKED FOR ---
    if os.path.exists(filename):
        print(f"\n>>> RESUMING FROM PREVIOUS BEST: {filename}")
        # Load the 278-score parameters
        best_params = np.load(filename)
        # Create a new population based on your previous winner
        population = np.array([polynomial_mutation(best_params.copy(), mutation_rate=0.1) for _ in range(population_size)])
        population[0] = best_params.copy() # Ensure the original winner stays in the group
        best_reward = evaluate_policy(best_params) # Re-verify the starting score
        print(f">>> STARTING REWARD: {best_reward:.2f}\n")
    else:
        print("\n>>> NO SAVED FILE: Starting fresh with random population\n")
        population = np.random.uniform(-1, 1, (population_size, gene_size))
        best_reward = -np.inf
        best_params = None

    for gen in range(num_generations):
        fitness = np.array([evaluate_policy(ind) for ind in population])
        
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_reward:
            best_reward = fitness[current_best_idx]
            best_params = population[current_best_idx].copy()
            np.save(filename, best_params) # Updates the winner
        
        print(f"Gen {gen+1}/{num_generations} | Best Overall: {best_reward:.2f}")

        # Evolution steps
        num_elites = max(2, int(population_size * 0.05))
        elites = population[fitness.argsort()[::-1][:num_elites]]
        new_population = list(elites)
        while len(new_population) < population_size:
            parents = elites[np.random.choice(num_elites, 2, replace=False)]
            child = simulated_binary_crossover(parents[0], parents[1])
            child = polynomial_mutation(child)
            new_population.append(child)
        population = np.array(new_population)

if __name__ == "__main__":
    train()