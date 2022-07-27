import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import swifter
import multiprocessing as mp
#from multiprocessing import freeze_support
#import gc

classified_holdout = pd.read_json("./predicted_colors.json")
classified_holdout["predicted_color_sets"] = classified_holdout["predicted_color_sets"].apply(lambda x: set(x))
classified_holdout.drop("Colors", axis=1, inplace=True)

database = json.load(open("./for_pipeline/training_holdout_data.json"))["training"]
database_df = pd.DataFrame()
database_df["known_color_sets"] = [set(x["d0_colors"] + x["d1_colors"]) for x in database]
database_df["SMILES"] = [x["smiles"] for x in database]
classified_holdout["cosines"] = [np.array(x) for x in cosine_similarity(np.matrix([x for x in classified_holdout["Spectrum"]], dtype=np.int32), np.matrix([x["spectrum"] for x in database], dtype=np.int32))]
classified_holdout.drop("Spectrum", axis=1, inplace=True)
print("Ranking")
classified_holdout["rank"] = classified_holdout.apply(lambda x: pd.Series(x["cosines"]).rank(ascending=False)[database_df["SMILES"] == x["SMILES"]].min()-1, axis=1)
print("Done")
genes = sorted(list(set([item for subset in classified_holdout["predicted_color_sets"] for item in subset])))
print("Collecting")
    #gc.collect()

    #try pyinstruments
def find_rank(row):
    filtered_pred_color_set = row["filtered_color_sets"]
    filtered_color_set_length = len(filtered_pred_color_set)
    color_scores = database_df["known_color_sets"].apply(lambda z: len(filtered_pred_color_set.intersection(z)) / filtered_color_set_length if filtered_color_set_length else 1)
    return np.multiply(color_scores, row["cosines"]).rank(ascending=False)[database_df.index[database_df["SMILES"] == row["SMILES"]]].min()-1

def energy_function(chromosome, solution_idx):
    print(chromosome, solution_idx)
    allowed_colors = set([color for allele_value, color in zip(chromosome, genes) if allele_value])
    classified_holdout["filtered_color_sets"] = classified_holdout["predicted_color_sets"].apply(lambda x: x.intersection(allowed_colors))
    classified_holdout["new_ranks"] = classified_holdout.apply(find_rank, axis=1)
    energy = np.sum(np.sign(classified_holdout["rank"] - classified_holdout["new_ranks"]))
    return -1*energy

import pygad
ga_instance = pygad.GA(num_generations=40, 
            num_parents_mating=5, 
            fitness_func=energy_function, 
            sol_per_pop=mp.cpu_count()-1, 
            num_genes = len(genes), 
            gene_type=int, 
            init_range_low=0, 
            init_range_high=2, 
            parent_selection_type="sss",
            keep_parents=-1, 
            crossover_type="single_point", 
            crossover_probability=.1, 
            mutation_type="random", 
            mutation_probability=.1, 
            mutation_percent_genes="default",
            random_mutation_min_val=0,
            random_mutation_max_val=2,
            allow_duplicate_genes=True,
            parallel_processing=["process", mp.cpu_count()-1]) 

ga_instance.run()
solution, fitness, solution_idx = ga_instance.best_solution()
    
print(solution)

