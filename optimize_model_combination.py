import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import swifter
from multiprocessing import freeze_support
import gc

def main():
    classified_holdout = pd.read_json("./predicted_colors.json")
    classified_holdout["predicted_color_sets"] = classified_holdout["predicted_color_sets"].apply(lambda x: set(x))
    classified_holdout.drop("Colors", axis=1, inplace=True)
    print(len(classified_holdout))
    database = json.load(open("./for_pipeline/training_holdout_data.json"))["training"]
    database_df = pd.DataFrame()
    database_df["known_color_sets"] = [set(x["d0_colors"] + x["d1_colors"]) for x in database]
    database_df["SMILES"] = [x["smiles"] for x in database]
    classified_holdout["cosines"] = [np.array(x) for x in cosine_similarity(np.matrix([x for x in classified_holdout["Spectrum"]], dtype=np.int32), np.matrix([x["spectrum"] for x in database], dtype=np.int32))]
    #cosines = cosine_similarity(np.matrix([x for x in classified_holdout["Spectrum"]], dtype=np.int32), np.matrix([x["spectrum"] for x in database], dtype=np.int32))
    classified_holdout.drop("Spectrum", axis=1, inplace=True)
    print(len(database_df))
    print("Ranking")
    classified_holdout["rank"] = classified_holdout.swifter.progress_bar(True).apply(lambda x: pd.Series(x["cosines"]).rank(ascending=False)[database_df["SMILES"] == x["SMILES"]].min()-1, axis=1)
    print("Done")
    genes = sorted(list(set([item for subset in classified_holdout["predicted_color_sets"] for item in subset])))
    print("Collecting")
    gc.collect()

    #try pyinstruments
    def find_rank(row):
        filtered_pred_color_set = row["filtered_color_sets"]
        filtered_color_set_length = len(filtered_pred_color_set)
        color_scores = database_df["known_color_sets"].apply(lambda z: len(filtered_pred_color_set.intersection(z)) / filtered_color_set_length if filtered_color_set_length else 1)
        return np.multiply(color_scores, row["cosines"]).rank(ascending=False)[database_df.index[database_df["SMILES"] == row["SMILES"]]].min()-1

    def energy_function(chromosome):
        #translate chromosome to allowed colors
        allowed_colors = set([color for allele_value, color in zip(chromosome, genes) if allele_value])
        # filter color sets in holdout
        classified_holdout["filtered_color_sets"] = classified_holdout["predicted_color_sets"].apply(lambda x: x.intersection(allowed_colors))
        # find new ranks
        classified_holdout["new_ranks"] = classified_holdout.apply(find_rank, axis=1)
        #classified_holdout["new_ranks"] = classified_holdout.apply(find_rank, axis=1)

        # calculate energy
        energy = np.sum(np.sign(classified_holdout["rank"] - classified_holdout["new_ranks"]))
        print("Collecting")
        gc.collect()
        print("Done Collecting")
        return energy

    test_chromosome = [0 for _ in genes]
    for i, g in enumerate(genes):
        print(g)
        new_chromosome = list(test_chromosome)
        new_chromosome[i] = 1
        x = energy_function(new_chromosome)
        print("\t", x)

    return 0

if __name__ == "__main__":
    freeze_support()
    main()

#import SAGA_optimize
#chromosome = [SAGA_optimize.ElementDescription(low=0, high=1, mutate="mutatePopulationRangedInteger") for _ in genes]
#saga = SAGA_optimize.SAGA(stepNumber=100, temperatureStepSize=100, startTemperature=0.5, alpha=1, direction=-1, energyCalculation=energy_function, crossoverRate=0.5, mutationRate=3, annealMutationRate=1, populationSize=20, elementDescriptions=chromosome)
#optimized_chromosome = saga.optimize()