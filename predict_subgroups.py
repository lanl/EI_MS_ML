import json
import pandas as pd
import os
import jsonpickle

holdout_df = pd.DataFrame()
holdout = json.load(open("./for_pipeline/training_holdout_data.json"))["holdout"]
holdout_spectra = [entry["spectrum"] for entry in holdout]
holdout_df["Spectrum"] = holdout_spectra
holdout_df["Name"] = [entry["Name"] for entry in holdout]
holdout_df["SMILES"] = [entry["smiles"] for entry in holdout]
holdout_df["Colors"] = [frozenset(entry["d1_colors"] + entry["d0_colors"]) for entry in holdout]

pred_colors = [set() for _ in holdout_spectra]
for file in os.listdir("./for_pipeline/models/"):
    model_dat = json.load(open("./for_pipeline/models/" + file))
    transformed_spectra = jsonpickle.decode(model_dat["transformer"]).transform(holdout_spectra)
    predictions = jsonpickle.decode(model_dat["model"]).predict(transformed_spectra)
    color = model_dat["color"]
    for i, value in enumerate(predictions):
        if value == 1:
            pred_colors[i].add(color)
        elif value == 0:
            pass
        else:
            raise Exception("invalid")
holdout_df["predicted_color_sets"] = [list(x) for x in pred_colors]

holdout_df.to_json("./predicted_colors.json")
