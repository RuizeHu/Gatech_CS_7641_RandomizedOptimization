import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time

fitness = mlrose.FourPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(
    length=60, fitness_fn=fitness, maximize=True, max_val=2)

# Define decay schedule
schedule = mlrose.ExpDecay()

# Define initial state
# n=30
#init_state = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
# n=60
init_state = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1])
# n=10
#init_state = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])

# Solve problem using random hill climnb
start = time.time()
best_state, best_fitness, fitness_curve1 = mlrose.random_hill_climb(problem, 
                                                      max_attempts=100, max_iters=1000,
                                                      init_state=init_state, random_state=1, curve=True)
end = time.time()
rhc_time = end - start

# Solve problem using simulated annealing
start = time.time()
best_state, best_fitness, fitness_curve2 = mlrose.simulated_annealing(problem, schedule=schedule,
                                                      max_attempts=100, max_iters=1000,
                                                      init_state=init_state, random_state=1, curve=True)
end = time.time()
sa_time = end - start

# Solve problem using genetic algorithm
start = time.time()
best_state, best_fitness, fitness_curve3 = mlrose.genetic_alg(problem, pop_size=2000, mutation_prob=0,
                                                      max_attempts=300, max_iters=1000,
                                                      random_state=1, curve=True)
end = time.time()
ga_time = end - start

# Solve problem using mimic
start = time.time()
best_state, best_fitness, fitness_curve4 = mlrose.mimic(problem, pop_size=1000, keep_pct=0.1, 
                                                      max_attempts=100, max_iters=1000,
                                                      random_state=1, curve=True)
end = time.time()
mimic_time = end - start

time = [rhc_time, sa_time, ga_time, mimic_time]
#print(best_state)

#print(best_fitness)

#print(fitness_curve)

print(time)

plt.plot(fitness_curve1, label='RHC')
plt.plot(fitness_curve2, label='SA')
plt.plot(fitness_curve3, label='GA')
plt.plot(fitness_curve4, label='MIMIC')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness score')
plt.title('Convergence Curve n=60')
plt.legend(loc='lower right')
plt.show()

# n=30
#[0.014468193054199219, 0.013213872909545898, 9.669710874557495, 44.23754405975342]
# n=60
#[0.0035402774810791016, 0.0338292121887207, 11.160180807113647, 176.23640966415405]
# n=10
#[0.0030298233032226562, 0.007817983627319336, 8.280635833740234, 4.477161169052124]
