import pickle

with open("coxph_kept_columns.pkl", "rb") as f:
    kept_columns = pickle.load(f)

print(len(kept_columns))
print(kept_columns)