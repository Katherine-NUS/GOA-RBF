"""
.. module:: heuristic_methods
   :synopsis: Heuristic optimization methods

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: heuristic_methods
:Author: David Eriksson <dme65@cornell.edu>

"""

from ..pySOT1.experimental_design import LatinHypercube, SymmetricLatinHypercube
import numpy as np
from scipy import stats
from scipy.spatial import distance

# from matplotlib import style
# import matplotlib.lines as  mlines
# import seaborn as sns
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm

class GeneticAlgorithm:
    """Genetic algorithm

    This is an implementation of the real-valued Genetic algorithm that is useful for optimizing
    on a surrogate model, but it can also be used on its own. The mutations are normally distributed
    perturbations, the selection mechanism is a tournament selection, and the crossover oepration is
    the standard linear combination taken at a randomly generated cutting point.

    The number of evaluations are popsize x ngen

    :param function: Function that can be used to evaluate the entire population. It needs to
        take an input of size nindividuals x nvariables and return a numpy.array of length
        nindividuals
    :type function: Object
    :param dim: Number of dimensions
    :type dim: int
    :param xlow: Lower variable bounds, of length dim
    :type xlow: numpy.array
    :param xup: Lower variable bounds, of length dim
    :type xup: numpy.array
    :param intvar: List of indices with the integer valued variables (e.g., [0, 1, 5])
    :type intvar: list
    :param popsize: Population size
    :type popsize: int
    :param ngen: Number of generations
    :type ngen: int
    :param start: Method for generating the initial population
    :type start: string
    :param proj_fun: Function that can project ONE infeasible individual onto the feasible region
    :type proj_fun: Object

    :ivar nvariables: Number of variables (dimensions) of the objective function
    :ivar nindividuals: population size
    :ivar lower_boundary: lower bounds for the optimization problem
    :ivar upper_boundary: upper bounds for the optimization problem
    :ivar integer_variables: List of variables that are integer valued
    :ivar start: Method for generating the initial population
    :ivar sigma: Perturbation radius. Each pertubation is N(0, sigma)
    :ivar p_mutation: Mutation probability (1/dim)
    :ivar tournament_size: Size of the tournament (5)
    :ivar p_cross: Cross-over probability (0.9)
    :ivar ngenerations: Number of generations
    :ivar function: Object that can be used to evaluate the objective function
    :ivar projfun: Function that can be used to project an individual onto the feasible region
    """

    def __init__(self, function, dim, xlow, xup, intvar=None, popsize=100, ngen=100, start="SLHD", projfun=None):
        self.nvariables = dim
        self.nindividuals = popsize + (popsize % 2)  # Make sure this is even
        self.lower_boundary = np.array(xlow)
        self.upper_boundary = np.array(xup)
        self.integer_variables = []
        if intvar is not None:
            self.integer_variables = np.array(intvar)
        self.start = start
        self.sigma = 0.2
        self.p_mutation = 1.0/dim
        self.tournament_size = 5
        self.p_cross = 0.9
        self.ngenerations = ngen
        self.function = function
        self.projfun = projfun

    def optimize(self):
        """Method used to run the Genetic algorithm

        :return: Returns the best individual and its function value
        :rtype: numpy.array, float
        """

        #  Initialize population
        if isinstance(self.start, np.ndarray):
            # if initial sampling size doesn't match the number of individuals and variable dimension, print error
            if self.start.shape[0] != self.nindividuals or self.start.shape[1] != self.nvariables:
                raise ValueError("Unknown method for generating the initial population")
            # if initial positions are outside the domain, print error
            # np.min(A, axis = 0) returns the minimum in all rows
            if (not all(np.min(self.start, axis=0) >= self.lower_boundary)) or \
                    (not all(np.max(self.start, axis=0) <= self.upper_boundary)):
                raise ValueError("Initial population is outside the domain")
            population = self.start
        elif self.start == "SLHD":
            exp_des = SymmetricLatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "LHD":
            exp_des = LatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "Random":
            population = self.lower_boundary + np.random.rand(self.nindividuals, self.nvariables) *\
                (self.upper_boundary - self.lower_boundary)
        else:
            raise ValueError("Unknown argument for initial population")

        new_population = []
        #  Round positions
        if len(self.integer_variables) > 0:
            new_population = np.copy(population)
            population[:, self.integer_variables] = np.round(population[:, self.integer_variables])
            for i in self.integer_variables:
                ind = np.where(population[:, i] < self.lower_boundary[i])
                population[ind, i] += 1
                ind = np.where(population[:, i] > self.upper_boundary[i])
                population[ind, i] -= 1

        #  Evaluate all individuals
        function_values = self.function(population)
        if len(function_values.shape) == 2:
            function_values = np.squeeze(np.asarray(function_values))

        # Save the best individual
        ind = np.argmin(function_values)
        best_individual = np.copy(population[ind, :])
        best_value = function_values[ind]

        if len(self.integer_variables) > 0:
            population = new_population

        # Main loop
        for ngen in range(self.ngenerations):
            # Do tournament selection to select the parents
            competitors = np.random.randint(0, self.nindividuals, (self.nindividuals, self.tournament_size))
            ind = np.argmin(function_values[competitors], axis=1)
            winner_indices = np.zeros(self.nindividuals, dtype=int)
            for i in range(self.tournament_size):  # This loop is short
                winner_indices[np.where(ind == i)] = competitors[np.where(ind == i), i]

            parent1 = population[winner_indices[0:self.nindividuals//2], :]
            parent2 = population[winner_indices[self.nindividuals//2:self.nindividuals], :]

            # Averaging Crossover
            cross = np.where(np.random.rand(self.nindividuals//2) < self.p_cross)[0]
            nn = len(cross)  # Number of crossovers
            alpha = np.random.rand(nn, 1)

            # Create the new chromosomes
            parent1_new = np.multiply(alpha, parent1[cross, :]) + np.multiply(1-alpha, parent2[cross, :])
            parent2_new = np.multiply(alpha, parent2[cross, :]) + np.multiply(1-alpha, parent1[cross, :])
            parent1[cross, :] = parent1_new
            parent2[cross, :] = parent2_new
            population = np.concatenate((parent1, parent2))

            # Apply mutation
            scale_factors = self.sigma * (self.upper_boundary - self.lower_boundary)  # Account for dimensions ranges
            perturbation = np.random.randn(self.nindividuals, self.nvariables)  # Generate perturbations
            perturbation = np.multiply(perturbation, scale_factors)  # Scale accordingly
            perturbation = np.multiply(perturbation, (np.random.rand(self.nindividuals,
                                                                     self.nvariables) < self.p_mutation))

            population += perturbation  # Add perturbation
            population = np.maximum(np.reshape(self.lower_boundary, (1, self.nvariables)), population)
            population = np.minimum(np.reshape(self.upper_boundary, (1, self.nvariables)), population)

            # Map to feasible region if method exists
            if self.projfun is not None:
                for i in range(self.nindividuals):
                    population[i, :] = self.projfun(population[i, :])

            # Round chromosomes
            new_population = []
            if len(self.integer_variables) > 0:
                new_population = np.copy(population)
                population[:, self.integer_variables] = np.round(population[:, self.integer_variables])
                for i in self.integer_variables:
                    ind = np.where(population[:, i] < self.lower_boundary[i])
                    population[ind, i] += 1
                    ind = np.where(population[:, i] > self.upper_boundary[i])
                    population[ind, i] -= 1

            # Keep the best individual
            population[0, :] = best_individual

            #  Evaluate all individuals
            function_values = self.function(population)
            if len(function_values.shape) == 2:
                function_values = np.squeeze(np.asarray(function_values))

            # Save the best individual
            ind = np.argmin(function_values)
            best_individual = np.copy(population[ind, :])
            best_value = function_values[ind]
            # print('current best = ',best_value)
            # Use the positions that are not rounded
            if len(self.integer_variables) > 0:
                population = new_population

        return best_individual, best_value




class MultimodalEDA:
    """Estimation of Distribution Algorithm

    This is an implementation of the Estimation of Distribution Algorithm for multimodal optimization.
    The code is based on the following paper:

    Yang Q, Chen W N, Li Y, et al. Multimodal estimation of distribution algorithms[J].
    IEEE transactions on cybernetics, 2016, 47(3): 636-650.

    """

    def __init__(self, function, dim, xlow, xup, intvar=None, popsize=100, ngen=100, start="SLHD", projfun=None):
        self.nvariables = dim
        self.nindividuals = popsize   # Make sure this is even
        self.lower_boundary = np.array(xlow)
        self.upper_boundary = np.array(xup)
        self.integer_variables = []
        if intvar is not None:
            self.integer_variables = np.array(intvar)
        self.start = start
        self.sigma = 0.2
        self.p_mutation = 1.0/dim
        self.tournament_size = 5
        self.p_cross = 0.9
        self.ngenerations = ngen
        self.function = function
        self.projfun = projfun
        self.population = []
        self.new_population = []
        self.population_values = np.inf*np.ones([self.nindividuals])
        self.best_individual = None
        self.best_value = np.inf
        self.ngroups=None
        self.ngroupsSet=[2,4,6,8,10]
        self.mu = []
        self.delta = []



    def Initialization(self):

        if isinstance(self.start, np.ndarray):
            # if initial sampling size doesn't match the number of individuals and variable dimension, print error
            if self.start.shape[0] != self.nindividuals or self.start.shape[1] != self.nvariables:
                raise ValueError("Unknown method for generating the initial population")
            # if initial positions are outside the domain, print error
            # np.min(A, axis = 0) returns the minimum in all rows
            if (not all(np.min(self.start, axis=0) >= self.lower_boundary)) or \
                    (not all(np.max(self.start, axis=0) <= self.upper_boundary)):
                raise ValueError("Initial population is outside the domain")
            self.population = self.start
        elif self.start == "SLHD":
            exp_des = SymmetricLatinHypercube(self.nvariables, self.nindividuals)
            self.population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "LHD":
            exp_des = LatinHypercube(self.nvariables, self.nindividuals)
            self.population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "Random":
            self.population = self.lower_boundary + np.random.rand(self.nindividuals, self.nvariables) *\
                (self.upper_boundary - self.lower_boundary)
        else:
            raise ValueError("Unknown argument for initial population")


        #  Round positions
        if len(self.integer_variables) > 0:
            self.new_population = np.copy(self.population)
            self.population[:, self.integer_variables] = np.round(self.population[:, self.integer_variables])
            for i in self.integer_variables:
                ind = np.where(self.population[:, i] < self.lower_boundary[i])
                self.population[ind, i] += 1
                ind = np.where(self.population[:, i] > self.upper_boundary[i])
                self.population[ind, i] -= 1

        #  Evaluate all individuals
        for k in range(self.nindividuals):
            self.population_values[k] = self.function(self.population[k, :])
        if len(self.population_values.shape) == 2:
            self.population_values = np.squeeze(np.asarray(self.population_values))

        # Save the best individual
        # ind = np.argmin(self.population_values)
        # self.best_individual = np.copy(self.population[ind, :])
        # self.best_value = function_values[ind]

        if len(self.integer_variables) > 0:
            self.population = self.new_population


    def Grouping(self):
        self.ngroups = self.ngroupsSet[np.random.randint(0, len(self.ngroupsSet))]
        number = int(np.floor(self.nindividuals/self.ngroups))
        groups_dict = dict()
        for i in range(self.ngroups):
            item = []
            if i < self.ngroups-1:
                for k in range(number):
                    item.append(i*number+k)
            else:
                for s in range((self.ngroups-1)*number, self.nindividuals):
                    item.append(s)
            groups_dict[i] = item
        return groups_dict

    def DistributionEstimation(self, groups_dict):
        nich_num = len(groups_dict)
        mu = np.zeros([nich_num, self.nvariables])
        delta = np.zeros([nich_num, self.nvariables])
        for i, item in groups_dict.items():
            nSamples = len(item)
            samples = np.zeros([nSamples, self.nvariables])
            for index, val in enumerate(item):
                samples[index, :] = self.population[val, :]
            mu[i, :] = np.mean(samples, axis=0)
            delta[i, :] = np.std(samples, axis=0)

        return mu, delta

    def Generate_offspring(self, groups_dict, mu, delta):

        def Random_vector_generating(mean, std):
            offspring=np.zeros([self.nvariables])
            if np.random.random() < 0.5:
                offspring = np.random.normal(loc=mean, scale=std, size=len(mean))
            else:
                offspring = stats.cauchy.rvs(loc=mean, scale=std, size=len(mean))
            return offspring
        nich_num = len(groups_dict)
        new_offspring = np.zeros([nich_num, self.nvariables])
        offspring_values = np.inf*np.ones([nich_num])
        for i in range(nich_num):
            new_offspring[i, :] = Random_vector_generating(mu[i, :], delta[i, :])

        new_offspring = np.maximum(np.reshape(self.lower_boundary, (1, self.nvariables)), new_offspring)
        new_offspring = np.minimum(np.reshape(self.upper_boundary, (1, self.nvariables)), new_offspring)

        #evaluation
        for k in range(nich_num):
            offspring_values[k] = self.function(new_offspring[k, :])
        if len(offspring_values.shape) == 2:
            offspring_values = np.squeeze(np.asarray(offspring_values))

        return new_offspring, offspring_values



    def local_search(self, groups_dict):
        nich_num = len(groups_dict)
        mu = np.zeros([nich_num, self.nvariables])
        delta = np.zeros([nich_num, self.nvariables])
        sigma = 1.0e-3
        delta = 1.0e-4*np.ones([self.nvariables])
        N = 3

        for i, item in groups_dict.items():
            nSamples = len(item)
            samples = np.zeros([nSamples, self.nvariables])
            F = np.zeros([nSamples])
            for index, val in enumerate(item):
                samples[index, :] = self.population[val, :]
                F[index] = self.population_values[val]

            F_max = np.max(F)
            F_min = np.min(F)
            P = np.zeros([nSamples])
            for index, val in enumerate(item):
                if F_max == F_min:
                    P[index] = (F[index] - F_min + singma) / (F_max - F_min + sigma)
                else:
                    P[index] = (F[index] - F_min) / (F_max - F_min)

            for index, val in enumerate(item):
                if np.random.rand() <= P[index]:
                    can = np.zeros([N, self.nvariables])
                    can_vals = np.zeros([N])
                    for h in range(N):
                        can[h] = np.random.normal(loc=self.population[val, :], scale=delta*np.ones([self.nvariables]), size=self.nvariables)
                        can_vals[h] = self.function(can[h, :])
                    if len(can_vals.shape) == 2:
                        can_vals = np.squeeze(np.asarray(can_vals))

                    for h in range(N):
                        if can_vals[h] < self.population_values[val]:
                            #print("************************Local search succeed !********************")
                            self.population_values[val] = can_vals[h]
                            self.population[val, :] = can[h, :]


    def Update(self, new_offspring, offspring_values):
        nich_num = new_offspring.shape[0]
        for i in range(nich_num):
            dis = np.zeros([self.nindividuals])
            for k in range(self.nindividuals):
                dis[k] = distance.euclidean(new_offspring[i, :], self.population[k, :])
            id = np.argmin(dis)
            if offspring_values[i] < self.population_values[id]:
                self.population_values[id] = offspring_values[i]
                self.population[id, :] = new_offspring[i, :]


    def optimize(self):
        """Method used to run the Genetic algorithm

        :return: Returns the best individual and its function value
        :rtype: numpy.array, float
        """

        #Step 1  Initialize population
        self.Initialization()


        # Main loop
        for ngen in range(self.ngenerations):
            if ngen % 100 == 0:
                print('Current generation: ', str(ngen))
            # Step 2 Dynamic grouping
            groups_dict = self.Grouping()

            #Step 3 Distribution estimation in each group
            mu, delta = self.DistributionEstimation(groups_dict)

            #Step 4 Generate offspring
            new_offspring, offspring_values = self.Generate_offspring(groups_dict, mu, delta)

            #Step 5 Update population
            self.Update(new_offspring, offspring_values)

            self.local_search(groups_dict)

        return self.population, self.population_values


if __name__ == "__main__":

    # Vectorized Ackley function in dim dimensions
    # def obj_function(x):
    #     return -20.0*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/dim)) - \
    #         np.exp(np.sum(np.cos(2.0*np.pi*x), axis=1)/dim) + 20 + np.exp(1)

    # ga = GeneticAlgorithm(obj_function, dim, -15*np.ones(dim), 20*np.ones(dim),
    #                       popsize=100, ngen=100, start="SLHD")
    # x_best, f_best = ga.optimize()

    # Print the best solution found
    # print("\nBest function value: {0}".format(f_best))
    # print("Best solution: {0}".format(x_best))

    #  Add constraint of unit-1 norm and supply projection method
    #
    # def projection(x):
    #     return x / np.linalg.norm(x)
    #
    # ga = GeneticAlgorithm(obj_function, dim, -1*np.ones(dim), 1*np.ones(dim), popsize=100,
    #                       ngen=100, start="SLHD", projfun=projection)
    # x_best, f_best = ga.optimize()
    #
    # # Print the best solution found
    # print("\nBest function value: {0}".format(f_best))
    # print("Best solution: {0}".format(x_best))
    # print("norm(x_best) = {0}".format(np.linalg.norm(x_best)))


    from MFB import*

    dim = 2
    data = MFO_1(dim=dim)

    EDA = MultimodalEDA(data.objfunction_HF, data.dim, data.xlow, data.xup,
                          popsize=100, ngen=2000, start="SLHD")
    x_, f_ = EDA.optimize()

    # Print the best solution found

    print("\n population after optimization: {0}".format(x_))
    print("\nFunction values: {0}".format(f_))

    # if dim == 2:
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     x1 = np.arange(data.xlow[0], data.xup[0], 0.005*(data.xup[0]-data.xlow[0]))
    #     x2 = np.arange(data.xlow[0], data.xup[0], 0.005*(data.xup[0]-data.xlow[0]))
    #
    #
    #     yy = np.zeros([x1.shape[0], x2.shape[0]])
    #     for i in range(x1.shape[0]):
    #         for k in range(x2.shape[0]):
    #             yy[i, k] = -data.objfunction_HF([x1[i], x2[k]])
    #     x1, x2 = np.meshgrid(x1, x2)
    #     ax.plot_surface(x1, x2, yy, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #     plt.xticks(np.arange(data.xlow[0], data.xup[0], 0.5))
    #     plt.yticks(np.arange(data.xlow[1], data.xup[1], 0.5))
    #     ax.set_zticks([-150, -50, 50])
    #     #ax.set_zlim([-100, 100])
    #
    #
    #     ax.scatter(x_[:,0], x_[:,1], -f_[:], c='k', s=80)
    #     plt.show()