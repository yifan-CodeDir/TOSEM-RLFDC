

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm

FL_DIR = f"./docker/results/localisation/"

# python3 summarize_tfd_FL_results.py origin_TS 120 TfD_network -q 10 -o ./output_analysis/origin_b120_q10_TfDnetwork_output.pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_suite_id', type=str)
    parser.add_argument('time_budget', type=int)
    parser.add_argument('metric', type=str)
    parser.add_argument('--max-query-budget', '-q', type=int, default=10)
    parser.add_argument('--output', '-o', type=str, default="./output.pkl")
    args = parser.parse_args()

    ts_id = args.test_suite_id
    time_budget = args.time_budget
    metric = args.metric
    query_budgets = list(range(args.max_query_budget + 1))
    result_dir = os.path.join(FL_DIR, ts_id)
    output_path = args.output

    rows = []
    for d4j_id in tqdm(os.listdir(result_dir), colour="green"):   # project-version folder
        if d4j_id.endswith(".pkl"):
            continue
        for initial_test_name in os.listdir(os.path.join(result_dir, d4j_id)):
            for rank_file_name in os.listdir(os.path.join(result_dir, d4j_id, initial_test_name)):
                # if filename.startswith("."):
                #     continue

                if f"ranks-{metric}-{args.max_query_budget}.pkl" not in rank_file_name:    ### FIXME: here we add "fair" to discriminate
                    continue

                groups = re.search("(\w+)-(\d+)-ranks", rank_file_name)
                if not groups:
                    continue
                project, version = groups.group(1), groups.group(2)
                # groups = re.search("noise_(\d\.\d)\.pkl", filename)
                # if groups:
                #     noise_prob = float(groups.group(1))
                #     if noise_prob not in noise_probs:
                #         continue
                # else:
                #     noise_prob = 0.0
                noise_prob = 0.0

                ranks = pd.read_pickle(
                    os.path.join(result_dir, d4j_id, initial_test_name, rank_file_name)
                )
                for query_budget in range(max(query_budgets) + 1):
                    if f"rank-{query_budget}" in ranks.columns:
                        buggy_ranks = ranks.loc[
                            ranks['is_buggy'] == True,
                            f"rank-{query_budget}"].values
                    for buggy_rank in buggy_ranks:  # record the ranks of all buggy elements
                        if query_budget in query_budgets:
                            rows.append(
                                [ts_id, project, version, initial_test_name, time_budget, query_budget, noise_prob, buggy_rank]
                            )

    df = pd.DataFrame(
        data=rows,
        columns=['Test Suite', 'Project', 'Version', "Initial Test", 
            'Time Budget', 'Query Budget', 'Noise Probability', 'Rank']
    )

    print(df)
    df.to_pickle(output_path)
    # df.to_csv(output_path)
    print(f"Saved to {output_path}")