import numpy as np
from deap import tools
from deap.algorithms import varAnd, varOr
import multiprocessing
from .callbacks.validations import eval_callbacks


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callbacks=None,
    verbose=True,
    estimator=None,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of Back2000.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    cxpb: float, default=None
        The probability of mating two individuals.

    mutpb: float, default=None
        The probability of mutating an individual.

    ngen: int, default=None
        The number of generation.

    stats: A :class:`~deap.tools.Statistics`
        Object that is updated inplace, optional.

    halloffame: A :class:`~deap.tools.HallOfFame`
        Object that will contain the best individuals, optional.

    callbacks: list or callable
        One or a list of the :class:`~sklearn_genetic.callbacks` methods available in the package.

    verbose: bool, default=True
        Whether or not to log the statistics.

    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """

    callbacks_start_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_start",
    }
    eval_callbacks(**callbacks_start_args)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    hof_size = len(halloffame.items) if (halloffame.items and estimator.elitism) else 0

    record = stats.compile(population) if stats else {}
    if isinstance(record["fitness"], np.ndarray):
        record = {key: value[0] for key, value in record.items()}

    n_gen = gen = 0
    logbook.record(gen=n_gen, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    # Check if any of the callbacks conditions are True to stop the iteration

    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    if eval_callbacks(**callbacks_step_args):
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        # Call ending callback
        eval_callbacks(**callbacks_end_args)
        print("INFO: Stopping the algorithm")
        return population, logbook, n_gen

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if estimator.elitism:
            offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if isinstance(record["fitness"], np.ndarray):
            record = {key: value[0] for key, value in record.items()}

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        callbacks_step_args = {
            "callbacks": callbacks,
            "record": record,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_step",
        }

        # Check if any of the callbacks conditions are True to stop the iteration
        if eval_callbacks(**callbacks_step_args):
            print("INFO: Stopping the algorithm")
            break

    n_gen = gen + 1

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_end",
    }

    # Call ending callback
    eval_callbacks(**callbacks_end_args)

    return population, logbook, n_gen


def eaMuPlusLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    callbacks=None,
    verbose=True,
    estimator=None,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    mu: int, default=None
        The number of individuals to select for the next generation.

    lambda\_: int, default=None
        The number of children to produce at each generation.

    cxpb: float, default=None
        The probability that an offspring is produced by crossover.

    mutpb: float, default=None
        The probability that an offspring is produced by mutation.

    ngen: int, default=None
        The number of generation.
    stats: A :class:`~deap.tools.Statistics`
        Object that is updated inplace, optional.

    halloffame: A :class:`~deap.tools.HallOfFame`
        Object that will contain the best individuals, optional.

    callbacks: list or Callable
        One or a list of the :class:`~sklearn_genetic.callbacks` methods available in the package.

    verbose: bool, default=True
        Whether or not to log the statistics.

    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """

    callbacks_start_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_start",
    }
    eval_callbacks(**callbacks_start_args)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    if isinstance(record["fitness"], np.ndarray):
        record = {key: value[0] for key, value in record.items()}

    n_gen = gen = 0
    logbook.record(gen=n_gen, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    # Check if any of the callbacks conditions are True to stop the iteration
    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    if eval_callbacks(**callbacks_step_args):
        # Call ending callback
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": None,
            "estimator": estimator,
            "method": "on_end",
        }

        eval_callbacks(**callbacks_end_args)
        print("INFO: Stopping the algorithm")
        return population, logbook, n_gen

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        if isinstance(record["fitness"], np.ndarray):
            record = {key: value[0] for key, value in record.items()}

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        callbacks_step_args = {
            "callbacks": callbacks,
            "record": record,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_step",
        }

        if eval_callbacks(**callbacks_step_args):
            print("INFO: Stopping the algorithm")
            break

    n_gen = gen + 1

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_end",
    }

    eval_callbacks(**callbacks_end_args)

    return population, logbook, n_gen


def eaMuCommaLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    n_jobs_ind_parallel,
    the_higher_metric_score_the_better,
    stats=None,
    halloffame=None,
    callbacks=None,
    verbose=True,
    estimator=None,
):
    """
    The base implementation is directly taken from: https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    This is the :math:`(\mu~,~\lambda)` evolutionary algorithm.

    population: A list of individuals.
        Population resulting of the iteration process.

    toolbox: A :class:`~deap.base.Toolbox`
        Contains the evolution operators.

    mu: int, default=None,
        The number of individuals to select for the next generation.

    lambda\_: int, default=None
        The number of children to produce at each generation.

    cxpb: float, default=None
        The probability that an offspring is produced by crossover.

    mutpb: float, default=None
        The probability that an offspring is produced by mutation.

    ngen: int, default=None
        The number of generation.

    stats: A :class:`~deap.tools.Statistics`
        Object that is updated inplace, optional.

    halloffame: A :class:`~deap.tools.HallOfFame`
        Object that will contain the best individuals, optional.

    callbacks: list or Callable
        One or a list of the :class:`~sklearn_genetic.callbacks` methods available in the package.

    verbose: bool, default=True
        Whether or not to log the statistics.

    estimator: :class:`~sklearn_genetic.GASearchCV`, default = None
        Estimator that is being optimized

    Returns
    -------

    pop: list
        The final population.

    log: Logbook
        Statistics of the evolution.

    n_gen: int
        Number of generations used.

    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    callbacks_start_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": None,
        "estimator": estimator,
        "method": "on_start",
    }

    eval_callbacks(**callbacks_start_args)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # Sequential Implementation

    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    # Parallel Implementation

    pool = multiprocessing.Pool(processes=n_jobs_ind_parallel)
    results = pool.map(toolbox.evaluate, invalid_ind)
    pool.close()
    pool.join()

    fitnesses = [result[0] for result in results]

    for _, current_generation_params in results:
        toolbox.log_results(current_generation_params)

    # End of Parallel Implementation

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    if isinstance(record["fitness"], np.ndarray):
        record = {key: value[0] for key, value in record.items()}

    n_gen = gen = 0
    logbook.record(gen=n_gen, nevals=len(invalid_ind), **record)

    if verbose:
        print(logbook.stream)

    # Additional Descriptor Function Calls

    compare_with_best_up_to_date(results, the_higher_metric_score_the_better)#M function to compare with best hyperparameters up to date
    
    print_best_hyperparameters(results) #M function to print out best hyperparameters for each generation. 

    # End of Additional Descriptor Function Calls

    callbacks_step_args = {
        "callbacks": callbacks,
        "record": record,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_step",
    }

    # Check if any of the callbacks conditions are True to stop the iteration
    if eval_callbacks(**callbacks_step_args):
        callbacks_end_args = {
            "callbacks": callbacks,
            "record": None,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_end",
        }

        eval_callbacks(**callbacks_end_args)
        print("INFO: Stopping the algorithm")
        return population, logbook, n_gen

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # Sequential Implementation

        #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        # Parallel Implementation

        pool = multiprocessing.Pool(processes=n_jobs_ind_parallel)
        results = pool.map(toolbox.evaluate, invalid_ind) #M Results is list of tuples,i.e, ([score], current_generation_params) from toolbox.evaluate
        pool.close()
        pool.join()

        fitnesses = [result[0] for result in results]

        #M Collate cv-scores of all individuals
        fitnesses = [result[0] for result in results]
        
        #M Record the results into the logbook sequentially
        for _, current_generation_params in results:
            toolbox.log_results(current_generation_params)

        # End of Parallel Implementation

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        if isinstance(record["fitness"], np.ndarray):
            record = {key: value[0] for key, value in record.items()}

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

        # Additional Descriptor Function Calls

        compare_with_best_up_to_date(results, the_higher_metric_score_the_better)#M function to compare with best hyperparameters up to date

        print_best_hyperparameters(results) #M function to print out best hyperparameters for each generation.

        # End of Additional Descriptor Function Calls

        callbacks_step_args = {
            "callbacks": callbacks,
            "record": record,
            "logbook": logbook,
            "estimator": estimator,
            "method": "on_step",
        }

        # Check if any of the callbacks conditions are True to stop the iteration
        if eval_callbacks(**callbacks_step_args):
            print("INFO: Stopping the algorithm")
            break

    n_gen = gen + 1

    callbacks_end_args = {
        "callbacks": callbacks,
        "record": None,
        "logbook": logbook,
        "estimator": estimator,
        "method": "on_end",
    }

    eval_callbacks(**callbacks_end_args)

    return population, logbook, n_gen

def compare_with_best_up_to_date(results,the_higher_metric_score_the_better):
    '''
    This method compares the best hyperparameters for each generation with the best hyperparameters obtained from previous generations
    '''
    global best_hyperparameters_up_to_date
    global best_fitness_score_up_to_date
    
    #M Unpack results
    scores, hyperparameters_profile_list = zip(*results)

    #M Find the index of the best score
    best_index = np.argmax(scores)
    best_score = scores[best_index]
    best_hyperparameters_profile = hyperparameters_profile_list[best_index]
    
    if not best_hyperparameters_up_to_date: #M if dictionary is empty(aka after evaluation of first generation)

        hyperparameters = extractHyperparametersfromBestProfile(best_hyperparameters_profile)
        
        #M Update up_to_date scores and hyperparameters
        best_fitness_score_up_to_date = best_score
        best_hyperparameters_up_to_date = hyperparameters
        
    else:
        if the_higher_metric_score_the_better:
            
            if best_score > best_fitness_score_up_to_date:
                
                hyperparameters = extractHyperparametersfromBestProfile(best_hyperparameters_profile)
                
                #M Update up_to_date scores and hyperparameters
                best_fitness_score_up_to_date = best_score
                best_hyperparameters_up_to_date = hyperparameters
            
        else:
            if best_score < best_fitness_score_up_to_date:
            
                hyperparameters = extractHyperparametersfromBestProfile(best_hyperparameters_profile)
            
                #M Update up_to_date scores and hyperparameters
                best_fitness_score_up_to_date = best_score
                best_hyperparameters_up_to_date = hyperparameters
            

def extractHyperparametersfromBestProfile(best_hyperparameters_profile):
    
        #M Extract only hyperparameter entries from the dictionary
        substring = 'modelForRxn'
        hyperparameter_keys = [key for key in best_hyperparameters_profile if substring in key]
        #hyperparameter_keys = [key for key in best_parameters if key.startswith('hybrid__modelForRxn')]
        hyperparameters = {key: best_hyperparameters_profile[key] for key in hyperparameter_keys}
        
        return hyperparameters
    
def print_best_hyperparameters(results):
    '''
    This method takes in the cv results of all individuals and prints the hyperparameter of the individual with the highest score
    '''
     #M Unpack results
    scores, parameters_list = zip(*results)

    #M Find the index of the best score
    best_index = np.argmax(scores)
    best_score = scores[best_index]
    best_parameters = parameters_list[best_index]
    
    #M Extract only hyperparameter entries from the dictionary
    hyperparameter_keys = [key for key in best_parameters if key.startswith('hybrid__modelForRxn')]
    hyperparameters = {key: best_parameters[key] for key in hyperparameter_keys}
    
    #M Print the best score
    print("Best Fitness Score:", best_score)
        
    #M Print best hyperparameters
    print("Best Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
        
    #M Print the best score up to date
    print("Best Fitness Score Up To Date:", best_fitness_score_up_to_date )
        
    #M Print best hyperparameters up to date
    print("Best Hyperparameters Up To Date:")
    for key, value in best_hyperparameters_up_to_date.items():
        print(f"{key}: {value}")
