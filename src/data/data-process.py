"""
Creates the base dataset by rearranging the correct answer to the first options
and rerunning the dataset through the LLM to isolate well understood questions.

This set will be used to than run the actual experiment around ordering.
"""

from dotenv import load_dotenv
import pandas as pd
from _dirs import *
from src.lib.process import *

load_dotenv()


def score_certainty(df, model):
    """ Get certainty score of a model for every dataset entry """
    df = df.copy()
    scores = score_dataset(df, model=model, times=4, randomize_options=True)
    # fill Nones and mark incorrect
    df["correctness"] = 0
    for i, df_score in enumerate(scores):
        # df_score["llm_answer_idx"] = df_score["llm_answer_idx"].fillna(-1)
        # df[f"llm_answer_idx_run_{i}"] = df_score["llm_answer_idx"]
        df[f"run_{i}_response_json"] = df_score["response_json"]
        df[f"run_{i}_response_idx"] = df_score["response_idx"]
        df[f"run_{i}_response"] = df_score["response"]
        df[f"run_{i}_options"] = df_score["options"]
        df_score["correctness"] = (df_score["response_idx"] == df_score["answer_idx"]).astype(int)
        df["correctness"] = df["correctness"] + df_score["correctness"]
    df["correctness"] = df["correctness"] / len(scores)
    # estimate consistency
    df['response_set'] = df.apply(lambda row: set([row[f"run_{i}_response"] for i,_ in enumerate(scores)]), axis=1)
    df["consistency"] = df['response_set'].apply(lambda x: 1 - (len(x)-1)/len(scores))
    # split into certain questions and uncertain
    print("high certainty base set")
    df["is_certain"] = (df["correctness"] == 1) & (df["consistency"] == 1)
    df_certain = df[df["is_certain"]]
    print("  count: ", len(df_certain.index))
    print("low certainty base set")
    df_uncertain = df[~df.index.isin(df_certain.index)]
    print("  count: ", len(df_uncertain.index))
    return df, scores


# Execute and add responses to DataFrame
if __name__ == "__main__":
    # cut 1000 random rows for experimentation
    print("cutting a test set for base scoring")
    df = pd.read_csv(DATA_DIR + '/mcq-all.csv.gz')\
            .sample(n=1000, random_state=1)\
            .reset_index(drop=True)
    print("  count: ", len(df.index))
    print("counts by source:")
    print(df.groupby("source").count())
    df.to_csv(DATA_DIR + '/mcq-1000.csv', index=False)

    # arrange answer to option 1 position
    df = df.apply(lambda r: relocate_answer(r, 0), axis=1)

    # gpt-3.5-turbo: determine certain vs uncertain questions
    print("scoring certainty for gpt-3.5-turbo")
    (df_gpt35, df_gpt35_runs) = score_certainty(df, model="gpt-3.5-turbo-0125")
    df_gpt35.to_csv(DATA_DIR + "/mcq-gpt-35-turbo.csv", index=False)
    for i, df_run in enumerate(df_gpt35_runs):
        df_run.to_csv(TMP_DIR + f"/mcq-gpt-35-turbo_run-{i}.csv", index=False)

    # gpt-40-mini: determine certain vs uncertain questions
    print("scoring certainty for gpt-4o-mini")
    (df_gpt40m, df_gpt40m_runs) = score_certainty(df, model="gpt-4o-mini")
    df_gpt40m.to_csv(DATA_DIR + "/mcq-gpt-4o-mini.csv", index=False)
    for i, df_run in enumerate(df_gpt40m_runs):
        df_run.to_csv(TMP_DIR + f"/mcq-gpt-4o-mini_run-{i}.csv", index=False)
