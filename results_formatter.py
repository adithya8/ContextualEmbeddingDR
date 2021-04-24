# This script formats the raw output of the DLATK to a formatted result csv file
# Can add support for generating graphs here

import pandas as pd
import numpy as np
from pprint import pprint 
import matplotlib.pyplot as plt 
import os
import sys


results_path_prefix = "/data/avirinchipur/ContextualEmbeddingDR3/results"
bs_sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
ks = [16, 32, 64, 128, 256, 512, 768]

'''
dr="rpca"
emb="bert"
outcomes=["age", "ext", "ope"]
table_name="T_20"
metric = "'r':"
'''
args = sys.argv
dr=args[1]
emb=args[2]
outcomes=eval(args[3])
table_name=args[4]
metric=args[5]
task_dir=args[6]

outcomes_combined="_".join(outcomes)
outcomes=sorted(outcomes)
file_names=[f"{dr}_{emb}_{k}_domain_bootstrap_{outcomes_combined}_{table_name}_nostd.txt" for k in ks]

full_result_table = {}
for outcome in (outcomes):
    full_result_table[outcome] = {}

for i, file_name in enumerate(file_names):
    if os.path.exists(results_path_prefix+f"/{task_dir}/"+file_name):
        f = open(results_path_prefix+f"/{task_dir}/"+file_name)
        result_text = f.readlines()
        filtered_result_text = [line.strip().split("(")[1].split(")")[0].split(",") for line in result_text if metric in line]
        filtered_result_text = [ np.around([float(text[0].strip()), float(text[1].strip())], 4) for text in filtered_result_text]
        
        for j in range(len(outcomes)):
            current_result = np.array(filtered_result_text[j*len(bs_sample_sizes): (j+1)*len(bs_sample_sizes)]).reshape(len(bs_sample_sizes), 2)
            full_result_table[outcomes[j]][ks[i]] = current_result
    else:
        continue
#pprint (full_result_table)


full_result_formatted = {}
for outcome in full_result_table:
    result_outcome = full_result_table[outcome]
    formatted_result_outcome = []
    for k in result_outcome:
        formatted_result_outcome.append(result_outcome[k].flatten().reshape(-1, 1))
    
    formatted_result_outcome = np.concatenate(formatted_result_outcome, axis=1)
    cols = [f"k={ks[k]}" for k in range(formatted_result_outcome.shape[1])]
    full_result_formatted[outcome] = pd.DataFrame(formatted_result_outcome, columns=cols)
    #print (f"outcome:{outcome}")
    #pprint (full_result_formatted[outcome])

f"{dr}_{emb}_{k}_domain_bootstrap_{outcomes_combined}_{table_name}_nostd.txt"
for outcome in outcomes:
    full_result_formatted[outcome].to_csv(f"formatted_results/{task_dir}/{dr}_{emb}_{outcome}_{table_name}.csv", index=False)
    
