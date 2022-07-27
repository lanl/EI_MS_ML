'''
Usage:
    main.py combine_MSPs <path_to_msp_directory>
'''

import os
import sqlite3
import requests
import pysmiles
import networkx as nx
import multiprocessing as mp
#import numpy as np
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
                            #print(entry)
                            entries.append(entry)
                            #if len(entries) == 10:
                            #    return entries
    print("Number of MSPs Combined: ", len(entries))
    if output:
        pass
        #todo need to write combined entries to json
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
            neighbor_list[n1].append((n2, ','+str(attr['order'])))
            neighbor_list[n2].append((n1, ','+str(attr['order'])))

    max_d = min(max_d, mol.number_of_nodes())
    colors = [[None for _ in range(mol.number_of_nodes())] for _ in range(max_d+1)]

    for node, d0_color in nx.get_node_attributes(mol, name="element").items():
        colors[0][node] = d0_color

    for current_d in range(max_d):
        current_colors = colors[current_d]
        for node, neighbors in neighbor_list.items():
            neighbor_colors = [f'({current_colors[neighbor]}{e})' for neighbor, e in neighbors]
            colors[current_d+1][node] = colors[0][node] + "(" + ','.join(sorted(neighbor_colors)) + ")"
    if all_colors:
        return {d: set(color_list) for d, color_list in enumerate(colors)}
    else:
        return colors[max_d]


def d_colorize_entries(entries, max_d=10):
    for entry in entries:
        if entry["smiles"]:
            entry["color_dict"] = d_colorize_entry(entry, all_colors=True, max_d=max_d)

def count_colors(entries):
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
    print("total unique colors")
    for depth, color_set in concatenated_dicts.items():
        print(depth, len(color_set))
    print(" > 1000 instances")
    for depth, color_set in concatenated_dicts.items():
        print(depth, len([x for x in color_set if counts[x] > 1000]))


def query_pubchem_for_smiles(inchikey, max_tries=10):
    fail_count = 0
    while fail_count < max_tries:
        try:
            r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON', timeout=5).json()
            if 'PropertyTable' in r:
                return r['PropertyTable']['Properties'][0]['CanonicalSMILES']
            elif 'Fault' in r:
                if r['Fault']['Message'] == 'No CID found':
                    return "no_match"
        #todo- yes, this is too generic of try-catch... deal with it.
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
                #todo - need to do something more here to handle this case?
    cur.close()
    conn.close()
    return entries

if __name__ == '__main__':
    mp.freeze_support()
    from docopt import docopt
    args = docopt(__doc__)
    print(args)

    if args['combine_MSPs'] and args['<path_to_msp_directory>']:
        entries = combine_NIST_MSPs(args['<path_to_msp_directory>'])
        add_smiles_to_entries(entries, inchikey_smiles_store="./inchikey_smiles_store.sqlite")
        #d_colorize_entries(entries, 1)
        #d_colorize_entries(entries, 2)
        d_colorize_entries(entries, 3)
        count_colors(entries)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
