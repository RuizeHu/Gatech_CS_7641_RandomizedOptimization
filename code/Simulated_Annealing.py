import mlrose
import numpy as np

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(
    length=8, fitness_fn=fitness, maximize=False, max_val=8)

# Define decay schedule
schedule = mlrose.ExpDecay()

# Define initial state
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# Solve problem using random hill climnb
#best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, 
#                                                      max_attempts=100, max_iters=1000,
#                                                      init_state=init_state, random_state=1, curve=True)

# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule=schedule,
                                                      max_attempts=100, max_iters=1000,
                                                      init_state=init_state, random_state=1, curve=True)

# Solve problem using genetic algorithm
#best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, pop_size=1000, mutation_prob=0.2,
#                                                      max_attempts=100, max_iters=1000,
#                                                      random_state=1, curve=True)

# Solve problem using genetic algorithm
#best_state, best_fitness, fitness_curve = mlrose.mimic(problem, pop_size=1000, keep_pct=0.2, 
#                                                      max_attempts=100, max_iters=1000,
#                                                      random_state=1, curve=True)

print(best_state)

print(best_fitness)

print(fitness_curve)
