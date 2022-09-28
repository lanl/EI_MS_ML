'''
Usage:
    optimize_model_combination.py <ga_wrapper_path> <time_limit_sec> <convergence_limit>
'''

# third-party libraries

import pygad
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# python built-in libraries

import warnings
import swifter
import multiprocessing as mp
import json
import sys
import time

# todo - yes those were globals and it needs to be refactored but pygad can't take nested functions
# todo - so that means you can't have a factory method to produce a properly scoped energy function


ga_wrapper = json.load(open(sys.argv[1]))
query_df = pd.read_pickle(ga_wrapper["query_df_path"])
reference_df = pd.read_pickle(ga_wrapper["reference_df_path"])
ref_true_color_matrix = np.matrix([x for x in reference_df["true_color_vector"]], dtype=np.uint8).T
colors = ga_wrapper["colors"]
query_df["indices"] = [reference_df["SMILES"] == qSMILES for qSMILES in query_df["SMILES"]]


start_time = time.time()
cluster_limit_hours = int(sys.argv[2])
convergence_limit = int(sys.argv[3])

def find_rank(row):
    if row["filtered_color_vector_magnitude"]:
        color_scores = np.dot(np.matrix(row["filtered_color_vector"], dtype=np.uint8), ref_true_color_matrix) / row["filtered_color_vector_magnitude"]
        return pd.Series(np.ravel(np.multiply(color_scores, row["cosines"]))).rank(ascending=False)[reference_df.index[row["indices"]]].min() - 1
    else:
        return row["orig_rank"]

def energy_function(chromosome, solution_idx):
    t1 = time.time()
    query_df["filtered_color_vector"] = query_df["pred_color_vector"].apply(lambda x: np.logical_and(x, chromosome))
    query_df["filtered_color_vector_magnitude"] = [np.sum(x) for x in query_df["filtered_color_vector"]]
    new_ranks = query_df.apply(find_rank, axis=1)
    diffs = query_df["orig_rank"] - new_ranks
    signs = np.sign(diffs)
    diffs_squared = diffs**2
    signed_squared_diffs = np.multiply(signs, diffs_squared)
    energy = np.sum(signed_squared_diffs)

    #energy = np.sum(np.sign(query_df["orig_rank"] - new_ranks))
    t2 = time.time()
    return energy

def on_generation_dump(ga_instance):
    current_time = time.time()
    print(current_time - start_time, cluster_limit_hours * 60 * 60)
    if current_time - start_time < cluster_limit_hours * 60 * 60:
        print("Total Pop Encountered: ", ga_wrapper["total_pop"], " Total Generations", ga_wrapper["total_generations"])
        best_chromosome, best_fitness, _ = ga_instance.best_solution()

        if "performance_history" in ga_wrapper:
            ga_wrapper["performance_history"].append(best_fitness)
        else:
            ga_wrapper["performance_history"] = [best_fitness]


        print("\t", best_chromosome, best_fitness)
        if best_fitness > 0 and (best_fitness == ga_wrapper["last_best_solution_fitness"] or best_chromosome.tolist() == ga_wrapper["last_best_solution_chromosome"]):
            ga_wrapper["converged"] += 1
            if ga_wrapper["converged"] == convergence_limit:
                print("Model Converged!")
                fitness = str(best_fitness)
                ga_wrapper["converged_solution"] = {
                    "total_generations": ga_wrapper["total_generations"],
                    "total_pop": ga_wrapper["total_pop"],
                    "converged_chromosome": best_chromosome.tolist()
                }
                ga_instance.save(ga_wrapper["ga_instance_path"] + "_converged_F=" + fitness)
                json.dump(ga_wrapper, open(ga_wrapper["wrapper_path"] + "_converged_F=" + fitness, 'w+'), indent=4, sort_keys=True)
        else:
            ga_wrapper["converged"] = 0
        chromosome_string = json.dumps(best_chromosome.tolist())
        if chromosome_string in ga_wrapper["all_solutions"]:
            ga_wrapper["all_solutions"][chromosome_string] = (chromosome_string, best_fitness, ga_wrapper["all_solutions"][chromosome_string][2]+1)
        else:
            ga_wrapper["all_solutions"][chromosome_string] = (chromosome_string, best_fitness, 1)
        ga_instance.save(ga_wrapper["ga_instance_path"])
        ga_wrapper["last_best_solution_chromosome"] = best_chromosome.tolist()
        ga_wrapper["last_best_solution_fitness"] = best_fitness
        ga_wrapper["total_pop"] += ga_instance.sol_per_pop
        ga_wrapper["total_generations"] += 1
        ga_wrapper["previous_population"] = ga_instance.population.tolist()
        json.dump(ga_wrapper, open(ga_wrapper["wrapper_path"], 'w+'), indent=4, sort_keys=True)
    else:
        print("call new instance")
        os.system("sbatch ./optimize_batch.sh " + ga_wrapper["wrapper_path"] + " " + str(cluster_limit_hours) + " " + str(convergence_limit))
        exit()


if __name__ == '__main__':
    if ga_wrapper["previous_population"]:
        previous_population = ga_wrapper["previous_population"]
    else:
        previous_population = None
    if ga_wrapper["converged"]:
        warnings.warn("Optimizing Converged Model")
    if ga_wrapper["total_generations"] >= 1000:
        print("generation limit met")
        exit()
    ga_instance = pygad.GA(
        num_generations=1000, #ga_wrapper["num_generations"],
        num_parents_mating=5,
        fitness_func=energy_function,
        sol_per_pop=40,
        num_genes=ga_wrapper["num_genes"],
        gene_type=int,
        init_range_low=ga_wrapper["init_range_low"],
        init_range_high=ga_wrapper["init_range_high"],
        parent_selection_type=ga_wrapper['parent_selection_type'],
        keep_parents=ga_wrapper['keep_parents'],
        crossover_type=ga_wrapper['crossover_type'],
        crossover_probability=ga_wrapper['crossover_probability'],
        mutation_type=ga_wrapper['mutation_type'],
        mutation_probability=ga_wrapper['mutation_probability'],
        mutation_percent_genes=ga_wrapper['mutation_percent_genes'],
        allow_duplicate_genes=ga_wrapper['allow_duplicate_genes'],
        parallel_processing=["process", mp.cpu_count() - 1],
        on_generation=on_generation_dump,
        gene_space=ga_wrapper['gene_space'],
        initial_population=previous_population
    )
    ga_instance.run()
