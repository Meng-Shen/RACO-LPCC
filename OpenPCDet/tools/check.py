import pandas as pd
import numpy as np

def load_drop(path):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        base = np.array([r["L0_Car_AP"], r["L0_Ped_AP"], r["L0_Cyc_AP"]])
        for l in range(1, 7):
            cur = np.array([r[f"L{l}_Car_AP"], r[f"L{l}_Ped_AP"], r[f"L{l}_Cyc_AP"]])
            rows.append(np.maximum(base - cur, 0))
    return np.stack(rows)

for name, path in [
    ("train", "split_AP_train.csv"),
    ("test", "test/split_AP.csv"),
]:
    x = load_drop(path)
    print(name)
    print("mean:", x.mean(axis=0))
    print("std :", x.std(axis=0))
    print("p50 :", np.percentile(x, 50, axis=0))
    print("p90 :", np.percentile(x, 90, axis=0))
    print("p99 :", np.percentile(x, 99, axis=0))