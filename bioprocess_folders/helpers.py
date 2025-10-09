# # Loading csv
# def load_ambr_csv(csv_file: str | Path) -> pd.DataFrame:
#     df = pd.read_csv(csv_file)
#     # Normalize time column name
#     time_candidates = ["time", "Time", "Time_min", "t", "t_min", "minutes"]
#     tcol = next((c for c in time_candidates if c in df.columns), None)
#     if tcol is None:
#         raise KeyError(f"No time column among {time_candidates}. Found: {list(df.columns)}")
#     df = df.rename(columns={tcol: "time"})
#     # Ensure starts at 0 (minutes)
#     t0 = float(df["time"].min())
#     df["time"] = df["time"] - t0
#     return df.sort_values("time").reset_index(drop=True)

# # visualize
# from pathlib import Path
# csv_path = Path("./experimental_dataset/ambr_run1_140323_19-24.csv")
# df_raw = load_ambr_csv(csv_path)
# df_raw.head()
# #--------------------------------------------------------------------------------

# # detecting bioreactors in the already-loaded file
# def detect_bioreactor_ids(columns):
#     ids = set()
#     pat = re.compile(r"^Bioreactor\s+(\d+)\s+-\s+")
#     for c in columns:
#         m = pat.match(c)
#         if m:
#             ids.add(int(m.group(1)))
#     return sorted(ids)

# # visualize what are the bioreactors available in the file
# rids = detect_bioreactor_ids(df_raw.columns)
# if not rids:
#     raise RuntimeError("No 'Bioreactor <ID> - <Signal>' columns found. Check CSV headers.")
# print("Detected reactor IDs:", rids)

# # selecting the bioreactor
# reactor_id = 19

# # ---> selector your observables yourself...


import pandas as pd
import re
from pathlib import Path

def load_ambr_csv(csv_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    time_candidates = ["time", "Time", "Time_min", "t", "t_min", "minutes"]
    tcol = next((c for c in time_candidates if c in df.columns), None)
    if tcol is None:
        raise KeyError(f"No time column among {time_candidates}. Found: {list(df.columns)}")
    df = df.rename(columns={tcol: "time"})

    t0 = float(df["time"].min())
    df["time"] = (df["time"] - t0)*60
    return df.sort_values("time").reset_index(drop=True)


csv_path = Path("./experimental_dataset/ambr_run1_140323_13-18.csv")
df_raw = load_ambr_csv(csv_path)

def detect_bioreactor_ids(columns):
    ids = set()
    pat = re.compile(r"^Bioreactor\s+(\d+)\s+-\s+")
    for c in columns:
        m = pat.match(c)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)

rids = detect_bioreactor_ids(df_raw.columns)
if not rids:
    raise RuntimeError("No 'Bioreactor <ID> - <Signal>' columns found. Check CSV headers.")
print("Detected reactor IDs:", rids)

reactor_id = 18
bioreactor_cols = [c for c in df_raw.columns if c.startswith(f"Bioreactor {reactor_id} -")]
df_single_bio_reactor = df_raw[["time"] + bioreactor_cols]
df_single_bio_reactor = df_single_bio_reactor.dropna(axis=1, how='all')

od_col = f"Bioreactor {reactor_id} - Optical density"
biomass_col = f"Bioreactor {reactor_id} - Biomass"
conversion_factor = 0.35

if od_col in df_single_bio_reactor.columns:
    df_single_bio_reactor = df_single_bio_reactor.dropna(subset=[od_col])
    df_single_bio_reactor[biomass_col] = df_single_bio_reactor[od_col] * conversion_factor
    df_single_bio_reactor = df_single_bio_reactor[["time", biomass_col]]
    df_single_bio_reactor = df_single_bio_reactor.iloc[:323,:]

else:
    print(f"Warning: {od_col} not found in columns. No OD filtering applied.")
# df_single_bio_reactor = df_single_bio_reactor.iloc[:498,:]

df_single_bio_reactor.to_csv(f"./experimental_dataset/bioreactor_{reactor_id}.csv", index=False)



