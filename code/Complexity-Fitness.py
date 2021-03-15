import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
import random

#fitness = mlrose.ContinuousPeaks(t_pct=0.1)
fitness = mlrose.FlipFlop()
bitlength = [i for i in range(10,100,10)]
schedule = mlrose.ExpDecay()
fitness_rhc_mean = []
fitness_sa_mean = []
fitness_ga_mean = []
fitness_mimic_mean = []
fitness_rhc_std = []
fitness_sa_std = []
fitness_ga_std = []
fitness_mimic_std = []


for i in bitlength:
    problem = mlrose.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True, max_val=2)

    fitness_rhc = []
    fitness_sa = []
    fitness_ga = []
    fitness_mimic = []
    for j in range(5):

        bitlist = []
        for k in range(0,i):
            x = random.randint(0, 1)
            bitlist.append(x)
    
        init_state = np.array(bitlist)

        best_state, best_fitness1, fitness_curve1 = mlrose.random_hill_climb(problem, 
                                                      max_attempts=400, max_iters=4000,
                                                      init_state=init_state, random_state=1, curve=True)
        fitness_rhc.append(best_fitness1)
    

        best_state, best_fitness2, fitness_curve2 = mlrose.simulated_annealing(problem, schedule=schedule,
                                                      max_attempts=400, max_iters=4000,
                                                      init_state=init_state, random_state=1, curve=True)
        fitness_sa.append(best_fitness2)
    

        best_state, best_fitness3, fitness_curve3 = mlrose.genetic_alg(problem, pop_size=2000, mutation_prob=0,
                                                      max_attempts=200, max_iters=1000,
                                                      random_state=1, curve=True)
        fitness_ga.append(best_fitness3)
    

        best_state, best_fitness4, fitness_curve4 = mlrose.mimic(problem, pop_size=2000, keep_pct=0.1, 
                                                      max_attempts=100, max_iters=1000,
                                                      random_state=1, curve=True)
        fitness_mimic.append(best_fitness4)

    fitness_rhc_mean.append(np.mean(fitness_rhc))
    fitness_sa_mean.append(np.mean(fitness_sa))
    fitness_ga_mean.append(np.mean(fitness_ga))
    fitness_mimic_mean.append(np.mean(fitness_mimic))
    fitness_rhc_std.append(np.std(fitness_rhc))
    fitness_sa_std.append(np.std(fitness_sa))
    fitness_ga_std.append(np.std(fitness_ga))
    fitness_mimic_std.append(np.std(fitness_mimic))

fitness_rhc_lb = [fitness_rhc_mean[i] - fitness_rhc_std[i] for i in range(len(bitlength))]
fitness_rhc_ub = [fitness_rhc_mean[i] + fitness_rhc_std[i] for i in range(len(bitlength))]
fitness_sa_lb = [fitness_sa_mean[i] - fitness_sa_std[i] for i in range(len(bitlength))]
fitness_sa_ub = [fitness_sa_mean[i] + fitness_sa_std[i] for i in range(len(bitlength))]
fitness_ga_lb = [fitness_ga_mean[i] - fitness_ga_std[i] for i in range(len(bitlength))]
fitness_ga_ub = [fitness_ga_mean[i] + fitness_ga_std[i] for i in range(len(bitlength))]
fitness_mimic_lb = [fitness_mimic_mean[i] - fitness_mimic_std[i] for i in range(len(bitlength))]
fitness_mimic_ub = [fitness_mimic_mean[i] + fitness_mimic_std[i] for i in range(len(bitlength))]
plt.fill_between(bitlength, fitness_rhc_lb,
                        fitness_rhc_ub, alpha=0.1, color='b')
plt.fill_between(bitlength, fitness_sa_lb,
                        fitness_sa_ub, alpha=0.1, color='r')
plt.fill_between(bitlength, fitness_ga_lb,
                        fitness_ga_ub, alpha=0.1, color='m')
plt.fill_between(bitlength, fitness_mimic_lb,
                        fitness_mimic_ub, alpha=0.1, color='k')
plt.plot(bitlength, fitness_rhc_mean, label='RHC', color='b')
plt.plot(bitlength, fitness_sa_mean, label='SA', color='r')
plt.plot(bitlength, fitness_ga_mean, label='GA', color='m')
plt.plot(bitlength, fitness_mimic_mean, label='MIMIC', color='k')
plt.xlabel('Length of bitstring')
plt.ylabel('Fitness score')
plt.title('Complexity-Fitness')
plt.legend(loc='lower right')
plt.show()


# n=30
#[0.014468193054199219, 0.013213872909545898, 9.669710874557495, 44.23754405975342]
# n=60
#[0.0035402774810791016, 0.0338292121887207, 11.160180807113647, 176.23640966415405]
# n=10
#[0.0030298233032226562, 0.007817983627319336, 8.280635833740234, 4.477161169052124]
