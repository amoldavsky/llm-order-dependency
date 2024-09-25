from _dirs import *
from src.lib.process import *
import pandas as pd


def reorder_and_score_dataset(df, model):
    df = df.copy()
    # shift the correct answer one option down one by one and re-score the dataset
    scores = []
    for i in range(4):
        print(f"scoring for position {i}")
        df_i = df.apply(lambda row: relocate_answer(row, i), axis=1)
        df_score = score_dataset(df_i, model=model, times=1)[0]
        scores.append(df_score)

    # process results
    for i, df_score in enumerate(scores):
        df[f"pos_{i}_response_json"] = df_score["response_json"]
        df[f"pos_{i}_response_proba"] = df_score["response_proba"]
        df[f"pos_{i}_response_idx"] = df_score["response_idx"]
        df[f"pos_{i}_response"] = df_score["response"]
        df[f"pos_{i}_options"] = df_score["options"]
        df[f"pos_{i}_correctness"] = (df_score["response_idx"] == df_score["answer_idx"]).astype(int)

    return df, scores


if __name__ == "__main__":
    # load base dataset for 3.5-turbo
    print("loading gpt-3.5 dataset")
    df_gpt35 = pd.read_csv(DATA_DIR + "/mcq-gpt-35-turbo.csv").sample(n=250, random_state=0)[[
        'question', 'options', 'answer', 'answer_idx', 'source',
        'correctness', 'consistency', 'is_certain', 'response_set'
    ]]
    print("  size: ", len(df_gpt35.index))
    print("certain vs uncertain questions: ")
    print(df_gpt35.groupby("is_certain").size())
    (df_gpt35_test, scores) = reorder_and_score_dataset(df_gpt35, "gpt-3.5-turbo-0125")
    df_gpt35_test.to_csv(DIST_DIR + "/mcq-gpt-35-turbo_test.csv", index=False)
    for i, df_run in enumerate(scores):
        df_run.to_csv(TMP_DIR + f"/gpt-35-turbo_test-pos-{i}.csv", index=False)

    # load base dataset for 4o-mini
    print("loading 4o-mini dataset")
    df_gpt4om = pd.read_csv(DATA_DIR + "/mcq-gpt-4o-mini.csv").sample(n=250, random_state=0)[[
        'question', 'options', 'answer', 'answer_idx', 'source',
        'correctness', 'consistency', 'is_certain', 'response_set'
    ]]
    print("  size: ", len(df_gpt4om.index))
    print("certain vs uncertain questions: ")
    print(df_gpt4om.groupby("is_certain").size())
    (df_gpt4om_test, scores) = reorder_and_score_dataset(df_gpt4om, "gpt-4o-mini")
    df_gpt4om_test.to_csv(DIST_DIR + "/mcq-gpt-4o-mini_test.csv", index=False)
    for i, df_run in enumerate(scores):
        df_run.to_csv(TMP_DIR + f"/gpt-4o-mini_test-pos-{i}.csv", index=False)
