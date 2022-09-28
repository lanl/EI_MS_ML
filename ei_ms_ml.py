'''
Usage:
    ei_ms_ml.py build_models <mainlib_path> <inchikey_smiles_store> <color_depth> <model_dir> [--tree_depth=<tree_depth>] [--num_trees=<num_trees>] [--seed=<seed>] [--min_color_count=<min_color_count>]
    ei_ms_ml.py prepare_for_pygad <mainlib_path> <inchikey_smiles_store> <color_depth> <model_dir> <reference_dump_path> <query_dump_path> <pygad_wrapper_path> <pygad_instance_path> [--tree_depth=<tree_depth>] [--num_trees=<num_trees>] [--seed=<seed>] [--min_color_count=<min_color_count>]
    ei_ms_ml.py evaluate_pygad_solution <reference_df_path> <replib_path> <inchikey_smiles_store> <color_depth> <model_dir> <tree_depth> <pygad_wrapper_path>
'''

# third-party libraries

import requests
import pysmiles
import networkx as nx
import numpy as np
import jsonpickle
import pandas as pd
import pygad
from docopt import docopt
from sklearn.feature_selection import SelectFwe
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity

# python built-in libraries
import os
import sqlite3
import multiprocessing as mp
import time
import random
import json
from collections import defaultdict


def combine_NIST_MSPs(path_to_msp_directory, output=None, mz_min=0, mz_max=500):
    entries = []
    for msp_filepath in [os.path.join(path_to_msp_directory, name) for name in os.listdir(path_to_msp_directory)]:
        with open(msp_filepath) as msp_filehandle:
            for msp_line in msp_filehandle:
                if msp_line.startswith("Name:"):
                    entry = {"Name": msp_line.lstrip("Name: ").rstrip()}
                elif msp_line.startswith("InChIKey: "):
                    entry["InChIKey"] = msp_line.lstrip("InChiKey: ").rstrip()
                elif msp_line.startswith("Num Peaks: "):
                    entry["NumPeaks"] = int(msp_line.lstrip("Num Peaks: ").rstrip())
                    entry["spectrum"] = [0 for _ in range(mz_max)]
                # todo: this logic could be tighter
                if "NumPeaks" in entry:
                    if msp_line != "\n":
                        for peak in msp_line.rstrip().split(";")[:-1]:
                            mz, intensity = peak.split()
                            mz = int(mz)
                            intensity = int(intensity)
                            if mz_min < mz < mz_max:
                                entry["spectrum"][mz] += intensity
                    else:
                        # print(entry)
                        entries.append(entry)
                        # if len(entries) == 10:
                        #    return entries
    print("Number of MSPs Combined: ", len(entries))
    if output:
        pass
        # todo need to write combined entries to json
    return entries


def d_colorize_entry(entry, max_d=1, all_colors=False, no_bond_order=False):
    try:
        mol = pysmiles.read_smiles(entry["smiles"])
    except:
        print("FAILURE")
        return None

    neighbor_list = defaultdict(list)
    for (n1, n2, attr) in nx.to_edgelist(mol):
        if no_bond_order:
            neighbor_list[n1].append((n2, ''))
            neighbor_list[n2].append((n1, ''))
        else:
            neighbor_list[n1].append((n2, ',' + str(attr['order'])))
            neighbor_list[n2].append((n1, ',' + str(attr['order'])))

    max_d = min(max_d, mol.number_of_nodes())
    colors = [[None for _ in range(mol.number_of_nodes())] for _ in range(max_d + 1)]

    for node, d0_color in nx.get_node_attributes(mol, name="element").items():
        colors[0][node] = d0_color

    for current_d in range(max_d):
        current_colors = colors[current_d]
        for node, neighbors in neighbor_list.items():
            neighbor_colors = [f'({current_colors[neighbor]}{e})' for neighbor, e in neighbors]
            colors[current_d + 1][node] = colors[0][node] + "(" + ','.join(sorted(neighbor_colors)) + ")"
    if all_colors:
        return {d: set(color_list) for d, color_list in enumerate(colors)}
    else:
        return colors[max_d]


def deduplicate_entries(entries, seed=None):
    if seed:
        random.seed(seed)
    random.shuffle(entries)
    unique_entries, duplicate_entries = [], []
    observed_smiles = set()
    for entry in entries:
        if entry['smiles'] in observed_smiles:
            duplicate_entries.append(entry)
        else:
            observed_smiles.add(entry['smiles'])
            unique_entries.append(entry)
    print("# Unique Entries: ", len(unique_entries), " # Duplicate Entries: ", len(duplicate_entries))
    return unique_entries, duplicate_entries


def d_colorize_entries(entries, max_d=10):
    new_entries = []
    for entry in entries:
        if entry["smiles"]:
            color_dict = d_colorize_entry(entry, all_colors=True, max_d=max_d)
            if color_dict:
                entry["color_dict"] = color_dict
                new_entries.append(entry)
    print("Entries Before Colorizing: ", len(entries), " Entries After Colorizing: ", len(new_entries))
    return new_entries


def summarize_results(predicted, ground_truth):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    mcc = matthews_corrcoef(predicted, ground_truth)

    for predicted, truth in zip(predicted, ground_truth):
        if predicted == truth and truth == 1:
            true_positive += 1
        elif predicted == truth and truth == 0:
            true_negative += 1
        elif predicted != truth and truth == 1:
            false_negative += 1
        else:
            false_positive += 1
    try:
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    except:
        accuracy = 0

    try:
        precision = true_positive / (true_positive + false_positive)
    except:
        precision = 0
    try:
        recall = true_positive / (true_positive + false_negative)
    except:
        recall = 0

    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'accuracy': accuracy
    }


def combine_summaries(summaries):
    temp_combined_summaries = {}
    for summary in summaries:
        for key, value in summary.items():
            if key not in temp_combined_summaries:
                temp_combined_summaries[key] = [value]
            else:
                temp_combined_summaries[key].append(value)
    return {key: np.mean(value) for key, value in temp_combined_summaries.items()}


def build_models(training_data, testing_data, hyperparam_dict=None, count_threshold=1000, max_depth=1, num_split=5,output_dir="./models/"):
    color_count_dict = count_colors(training_data)
    default_hyperparam_dict = {'num_trees': 100, 'depth': 10, 'n_jobs': mp.cpu_count() - 1}
    if hyperparam_dict:
        for key, value in hyperparam_dict.items():
            default_hyperparam_dict[key] = value
    hyperparam_dict = default_hyperparam_dict
    clf = RandomForestClassifier(
        n_jobs=hyperparam_dict['n_jobs'],
        class_weight="balanced_subsample",
        n_estimators=hyperparam_dict["num_trees"],
        max_depth=hyperparam_dict["depth"]
    )
    sss = StratifiedKFold(n_splits=num_split)
    spectra = np.array([entry["spectrum"] for entry in training_data], dtype=np.int64)

    for entry in training_data:
        entry["color_set"] = set()
        for depth, colors in entry['color_dict'].items():
            if depth <= max_depth:
                entry["color_set"].update(colors)
    num_models = 0
    for color, count in color_count_dict.items():
        print(color, count)
        model_results = {'color': color,
                         'count': count,
                         'hyperparameters': hyperparam_dict}
        color_vector = np.array([1 if color in entry["color_set"] else 0 for entry in training_data], dtype=np.int8)
        if count > count_threshold and np.sum(color_vector) > num_split:
            num_models += 1
            performance_summaries = []
            for train_index, test_index in sss.split(spectra, color_vector):
                spectra_train_split, color_vector_train_split = spectra[train_index], color_vector[train_index]
                spectra_test_split, color_vector_test_split = spectra[test_index], color_vector[test_index]
                transformer = SelectFwe(alpha=.05 / spectra.shape[1])
                transformer.fit(spectra_train_split, color_vector_train_split)
                spectra_train_split = transformer.transform(spectra_train_split)
                spectra_test_split = transformer.transform(spectra_test_split)
                clf.fit(spectra_train_split, color_vector_train_split)
                performance_summaries.append(
                    summarize_results(clf.predict(spectra_test_split), color_vector_test_split))
            model_results['performance'] = combine_summaries(performance_summaries)
            print("\t", model_results['performance'])
            transformer = SelectFwe(alpha=.05 / spectra.shape[1])
            transformer.fit(spectra, color_vector)
            transformed_spectra = transformer.transform(spectra)
            model_results['model'] = jsonpickle.encode(clf.fit(transformed_spectra, color_vector))
            model_results['transformer'] = jsonpickle.encode(transformer)
            json.dump(model_results, open(output_dir + color + "_" + str(max_depth) + "c_" + str(hyperparam_dict["depth"]) + "d.json", 'w+'))
    print("# Models Built: ", num_models)


def count_colors(entries, print_to_shell=False):
    concatenated_dicts = {}
    counts = {}
    for entry in entries:
        if "color_dict" in entry:
            try:
                for depth, color_set in entry["color_dict"].items():
                    if depth in concatenated_dicts:
                        concatenated_dicts[depth].update(color_set)
                    else:
                        concatenated_dicts[depth] = color_set
                    for color in color_set:
                        if color in counts:
                            counts[color] += 1
                        else:
                            counts[color] = 1
            except:
                pass
    if print_to_shell:
        print("total unique colors")
        for depth, color_set in concatenated_dicts.items():
            print(depth, len(color_set))
        print(" > 1000 instances")
        for depth, color_set in concatenated_dicts.items():
            print(depth, len([x for x in color_set if counts[x] > 1000]))
    return counts


def query_pubchem_for_smiles(inchikey, max_tries=10):
    fail_count = 0
    while fail_count < max_tries:
        try:
            r = requests.get(
                f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON',
                timeout=5).json()
            if 'PropertyTable' in r:
                return r['PropertyTable']['Properties'][0]['CanonicalSMILES']
            elif 'Fault' in r:
                if r['Fault']['Message'] == 'No CID found':
                    return "no_match"
        # todo- yes, this is too generic of try-catch... deal with it.
        except:
            fail_count += 1
            if fail_count & 5 == 0:
                time.sleep(5)
    return False


def add_smiles_to_entries(entries, inchikey_smiles_store=None):
    conn = sqlite3.connect(inchikey_smiles_store)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS entries (inchikey text, smiles text)")
    cur.execute("CREATE INDEX IF NOT EXISTS inchikey_index ON entries(inchikey)")
    conn.commit()
    none_count = 0
    for entry in entries:
        results = cur.execute("SELECT smiles FROM entries where inchikey=?", (entry["InChIKey"],)).fetchall()
        if results:
            smiles = results[0][0]
            if smiles == 'no_match':
                entry["smiles"] = None
                none_count += 1
            else:
                entry["smiles"] = results[0][0]
        else:
            smiles = query_pubchem_for_smiles(entry["InChIKey"])
            if smiles:
                entry["smiles"] = smiles
                cur.execute("INSERT INTO entries VALUES (?,?)", (entry["InChIKey"], smiles))
                conn.commit()
            else:
                entry["smiles"] = None
                none_count += 1
                # todo - need to do something more here to handle this case?
    cur.close()
    conn.close()
    return entries


def predict_subgroups(entries, model_directory, depth=30):
    spectra_for_predictions = [e["spectrum"] for e in entries]
    predicted_colors = [set() for _ in entries]
    for model in os.listdir(model_directory):
        if str(depth) in model:
            print("Predicting: ", model)
            try:
                model_dict = json.load(open(model_directory + "/" + model))
                model_dict["transformer"] = jsonpickle.decode(model_dict["transformer"])
                model_dict["model"] = jsonpickle.decode(model_dict["model"])
                predictions = model_dict["model"].predict(model_dict["transformer"].transform(spectra_for_predictions))
                for i, pred in enumerate(predictions):
                    if pred:
                        predicted_colors[i].add(model_dict["color"])
            except:
                pass
        else:
            print("Skipping: ", model)
    for predicted_color_set, entry in zip(predicted_colors, entries):
        entry["predicted_color_set"] = predicted_color_set
    return entries


def extract_relevant_colors(entries, model_directory, depth=30):
    allowed_colors = set()
    for model in os.listdir(model_directory):
        #print(model)
        if str(depth) in model:
            model_dict = json.load(open(model_directory + "/" + model))
            allowed_colors.add(model_dict["color"])
    for entry in entries:
        #print(entry)
        nascent_entry_colors = set()
        for color_set in entry["color_dict"].values():
            nascent_entry_colors.update(color_set)
        entry["all_colors"] = nascent_entry_colors.intersection(allowed_colors)
    return entries


def convert_to_dataframe(entries):
    dataframe = pd.DataFrame()
    dataframe["predicted_color_sets"] = [entry["predicted_color_set"] for entry in entries if "predicted_color_set" in entry]
    dataframe["true_color_sets"] = [entry["all_colors"] for entry in entries]
    dataframe["spectra"] = [entry["spectrum"] for entry in entries]
    dataframe["SMILES"] = [entry["smiles"] for entry in entries]
    dataframe.infer_objects()
    return dataframe


def dump_for_optimization(reference_df, query_df, reference_dump_path, query_dump_path):
    query_df["cosines"] = [np.array(x) for x in cosine_similarity(np.matrix([x for x in query_df["spectra"]], dtype=np.int32), np.matrix([x for x in reference_df["spectra"]], dtype=np.int32))]
    query_df["orig_rank"] = query_df.apply(lambda x: pd.Series(x["cosines"]).rank(ascending=False)[reference_df["SMILES"] == x["SMILES"]].min() - 1, axis=1)
    #query_df.drop("spectra", axis=1, inplace=True)
    #reference_df.drop("spectra", axis=1, inplace=True)

    colors = sorted(list(set([c for s in query_df["predicted_color_sets"] for c in s])))

    pred_color_vectors = []
    for z in query_df["predicted_color_sets"]:
        color_vector = [1 if x in z else 0 for x in colors]
        pred_color_vectors.append(np.array(color_vector, dtype=np.int8))
    query_df["pred_color_vector"] = pred_color_vectors

    true_color_vectors = []
    for z in reference_df["true_color_sets"]:
        color_vector = [1 if x in z else 0 for x in colors]
        true_color_vectors.append(np.array(color_vector, dtype=np.int8))
    reference_df["true_color_vector"] = true_color_vectors

    reference_df.to_pickle(reference_dump_path)
    query_df.to_pickle(query_dump_path)

    ga_wrapper = {
        "wrapper_path": arguments["<pygad_wrapper_path>"],
        "query_df_path": query_dump_path,
        "reference_df_path": reference_dump_path,
        "colors": colors,
        "ga_instance": None,
        "ga_instance_path": arguments['<pygad_instance_path>'],
        "previous_ga_instance": None,
        "total_generations": 0,
        "total_pop": 0,
        "last_best_solution_chromosome": None,
        "last_best_solution_fitness": None,
        "previous_population": None,
        "num_generations": 2,
        "num_parents_mating": 5,
        "num_genes": len(colors),
        "init_range_low": 0,
        "init_range_high": 2,
        "parent_selection_type": 'sss',
        "keep_parents": -1,
        "crossover_type": 'single_point',
        "crossover_probability": .1,
        "mutation_type": 'random',
        "mutation_percent_genes": "default",
        "mutation_probability": .1,
        "allow_duplicate_genes": True,
        "gene_space": [0, 1],
        "converged": 0,
        "all_solutions": {}
    }
    json.dump(ga_wrapper, open(ga_wrapper["wrapper_path"], 'w+'), indent=4, sort_keys=True)


def find_rank_factory(reference_df):
    def find_rank(row):
        filtered_pred_color_set = row["filtered_color_sets"]
        filtered_color_set_length = len(filtered_pred_color_set)
        color_scores = reference_df["true_color_sets"].apply(lambda z: len(
            filtered_pred_color_set.intersection(z)) / filtered_color_set_length if filtered_color_set_length else 1)
        return np.multiply(color_scores, row["cosines"]).rank(ascending=False)[
                   reference_df.index[reference_df["SMILES"] == row["SMILES"]]].min() - 1

    return find_rank


def evaluate_pygad_solution(solution, colors, reference_df, query_df):
    query_df["cosines"] = [np.array(x) for x in cosine_similarity(np.matrix([y for y in query_df["spectra"]], dtype=np.int32), np.matrix([z for z in reference_df["spectra"]], dtype=np.int32))]
    query_df["orig_rank"] = query_df.apply(lambda x: pd.Series(x["cosines"], dtype=np.float32).rank(ascending=False)[reference_df["SMILES"] == x["SMILES"]].min() - 1, axis=1)
    allowed_colors = set([color for allele_value, color in zip(solution, colors) if allele_value])
    query_df["filtered_color_sets"] = query_df["predicted_color_sets"].apply(lambda x: x.intersection(allowed_colors))
    find_rank = find_rank_factory(reference_df)
    new_ranks = query_df.apply(find_rank, axis=1)
    num_improved = np.sum(np.sign(query_df["orig_rank"] - new_ranks))
    print("Solution improved :", num_improved)
    json.dump({"original_ranks": [x for x in query_df["orig_rank"]], "new_ranks": [x for x in new_ranks]}, open("replib_improvement.json", 'w+'))
    return num_improved


if __name__ == '__main__':
    mp.freeze_support()
    arguments = docopt(__doc__)
    print(arguments)
    if arguments['prepare_for_pygad']:
        print("Combining")
        mainlib = combine_NIST_MSPs(arguments['<mainlib_path>'])
        print("Smiling")

        os.system("cp " + arguments["<inchikey_smiles_store>"] + " /tmp/inchikey_colors_store.sqlite")
        

        mainlib = add_smiles_to_entries(mainlib, inchikey_smiles_store="/tmp/inchikey_colors_store.sqlite")
        print("Coloring")
        mainlib = d_colorize_entries(mainlib, int(arguments['<color_depth>']))
        if arguments['--seed']:
            training_data, testing_data = deduplicate_entries(mainlib, int(arguments['--seed']))
        else:
            training_data, testing_data = deduplicate_entries(mainlib)
        hyperparam_dict = {}
        if arguments['--min_color_count']:
            count_threshold = int(arguments['--min_color_count'])
        else:
            count_threshold = 1000

        if arguments['--num_trees']:
            num_trees = int(arguments['--num_trees'])
        else:
            num_trees = 100

        if arguments['--tree_depth']:
            tree_depth = int(arguments['--tree_depth'])
        else:
            tree_depth = 30
        print("Building Models")
        #build_models(training_data, testing_data, hyperparam_dict={'n_estimators': num_trees, 'depth': tree_depth}, count_threshold=count_threshold, max_depth=int(arguments['<color_depth>']), num_split=5, output_dir=arguments['<model_dir>'])
        print("Extracting")
        training_data = extract_relevant_colors(training_data, arguments['<model_dir>'], tree_depth)
        testing_data = extract_relevant_colors(testing_data, arguments['<model_dir>'], tree_depth)
        print("Predicting")
        testing_data = predict_subgroups(testing_data, arguments['<model_dir>'], tree_depth)
        print("Converting")
        training_df = convert_to_dataframe(training_data)
        testing_df = convert_to_dataframe(testing_data)
        print("Dumping")
        dump_for_optimization(training_df, testing_df, arguments['<reference_dump_path>'], arguments['<query_dump_path>'])
    elif arguments['build_models']:
        mainlib = combine_NIST_MSPs(arguments['<mainlib_path>'])
        mainlib = add_smiles_to_entries(mainlib, inchikey_smiles_store=arguments["<inchikey_smiles_store>"])
        mainlib = d_colorize_entries(mainlib, int(arguments['<color_depth>']))
        if arguments['--seed']:
            training_data, testing_data = deduplicate_entries(mainlib, int(arguments['--seed']))
        else:
            training_data, testing_data = deduplicate_entries(mainlib)
        hyperparam_dict = {}
        if arguments['--min_color_count']:
            count_threshold = int(arguments['--min_color_count'])
        else:
            count_threshold = 1000

        if arguments['--num_trees']:
            num_trees = int(arguments['--num_trees'])
        else:
            num_trees = 1000

        if arguments['--tree_depth']:
            tree_depth = int(arguments['--tree_depth'])
        else:
            tree_depth = 30
        #build_models(training_data, testing_data, hyperparam_dict={'n_estimators': num_trees, 'depth': tree_depth},
        #             count_threshold=count_threshold, max_depth=int(arguments['<color_depth>']), num_split=5,
        #             output_dir=arguments['<model_dir>'])
    elif arguments['evaluate_pygad_solution']:
        wrapper = json.load(open(arguments["<pygad_wrapper_path>"]))
        solution = wrapper["converged_solution"]["converged_chromosome"]
        colors = wrapper["colors"]
        replib = combine_NIST_MSPs(arguments['<replib_path>'])
        replib = add_smiles_to_entries(replib, arguments['<inchikey_smiles_store>'])
        replib = d_colorize_entries(replib, int(arguments['<color_depth>']))
        replib = predict_subgroups(replib, arguments['<model_dir>'], arguments['<tree_depth>'])
        replib = extract_relevant_colors(replib, arguments['<model_dir>'], arguments['<tree_depth>'])
        replib_df = convert_to_dataframe(replib)
        reference_df = pd.read_pickle(arguments['<reference_df_path>'])
        evaluate_pygad_solution(solution, colors, reference_df, replib_df)
