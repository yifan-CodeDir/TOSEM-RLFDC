import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()

    excluded_dict = {
        # 'Lang': [2, 12,23, 56, 63,65],
        'Lang': [2, 12,23, 56, 65,10,30,41],  
        'Chart': [4],  
        # 'Chart': [],
        'Time': [5, 21, 22],  # FIXME: remove 27    
        # 'Time': [],
        # 'Math': [17, 18, 19, 20, 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,63, 80,81,98,100,101,102],
        'Math': [15,16,17, 18, 19, 20, 21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,80,81,98,100,101,102,63,54,59],
        'Closure': [63, 93,1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 28, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 105, 107, 112, 116, 117, 118, 123, 125, 127, 129]      
    }
    # Lang:57, Chart:25, Time:24, Math:76, Closure:66, Total: 248
    # Lang:62, Chart:25, Time:25, Math:83, Closure:109
    df = pd.read_pickle(args.file)
    query_budgets = [1,2,3,4,5,6,7,8,9,10]
    N = [1,3,5,10]

    df["Rank"] = df["Rank"].astype(np.float64)
    df["Version"] = df["Version"].astype(np.int32)

    # exclude some program
    # count = 0
    for p in excluded_dict: 
        excluded_list = excluded_dict[p]
        df = df[~((df["Project"].isin([p])) & (df["Version"].isin(excluded_list)))]
        # print(len(excluded_dict[p]))
        # count += len(excluded_dict[p])
    # print(count)

    tdf = df[df["Query Budget"].isin(query_budgets)]
    tdf = tdf[tdf["Noise Probability"] == 0.0]
    # get higest rank amont all buggy elements
    mdf = tdf.groupby(["Test Suite", "Time Budget", "Query Budget",
        "Project", "Version", "Initial Test"]).min()
    # add helper columns to calculate acc@n
    for n in N:
        mdf[f"acc@{n}"] = (mdf["Rank"] <= n).astype(float)
    mdf.reset_index(inplace=True)
    num_subjects = mdf[["Project", "Version"]].drop_duplicates().shape[0]
    print(f"# Subjects: {num_subjects}")

    # mdf = mdf.groupby(["Test Suite", "Time Budget", "Query Budget",
    #     "Project", "Version"]).mean()  # calculate average for multiple failing tests

    # Calculate the averaged acc@n over all test suite
    acc_n_for_each_bug = mdf.groupby(["Test Suite", "Time Budget",
        "Query Budget", "Project", "Version"]).mean()[[f"acc@{n}" for n in N]]

    # acc_n_for_each_subject = acc_n_for_each_bug.groupby(["Test Suite", "Time Budget",
    #     "Query Budget", "Project"]).sum()[[f"acc@{n}" for n in N]]
    # acc_n_for_each_subject = acc_n_for_each_subject.reset_index().round(0)
    # print(acc_n_for_each_subject)

    acc_n = acc_n_for_each_bug.groupby(
        ["Test Suite", "Time Budget", "Query Budget"]).sum()  # sum for all programs
    acc_n = acc_n.reset_index().round(0)

    print(acc_n)
