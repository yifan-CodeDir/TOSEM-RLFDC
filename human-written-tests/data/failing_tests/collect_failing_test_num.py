import os
import pandas as pd

excluded = {
    'Lang': [2, 23, 56],
    'Chart': [],
    'Time': [21],
    'Math': [],
    'Closure': [63, 93]      
}
projects = {
    'Lang':    (1, 65),
    # 'Lang':    (1, 1),
    'Chart':   (1, 26),
    'Time':    (1, 27),
    'Math':    (1, 106),
    'Closure': (1, 133)
}

record_df = pd.DataFrame(columns=["project","version","num_total_failing_tests"])

for p in projects:
    start, end = projects[p]
    for v in range(start, end + 1):
        if v in excluded[p]:
            continue
        with open("./{}/{}".format(p, v), "r") as f:
            record_df.loc[len(record_df.index)] = [p, v, len(f.readlines())]

record_df.to_csv("real_failing_test_num.csv")
record_df.drop(["version"], axis=1, inplace=True)
print(record_df.groupby(["project"]).sum())