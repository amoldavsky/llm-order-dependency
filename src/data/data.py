"""
Preps testing data by fetches MCQ dataset from various sources - commonsence, race, trivia, arc.
"""

import pandas as pd
from datasets import load_dataset
import random
from _dirs import *

# Define output directory
OUTPUT_DIR = TMP_DIR + "/mcq_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_race():
    """
    Load the RACE dataset from HuggingFace.
    Returns a pandas DataFrame with standardized columns.
    """
    print("Fetching RACE dataset...")
    df_race = pd.DataFrame(
        load_dataset(
            "race",
            "middle",
            split="train"
        )
    ).sample(frac=1).head(1000).reset_index(drop=True)
    def extract_options_and_answer(row):
        options = row['options']
        answer_latter = row['answer'].lower()
        answer_idx = ord(answer_latter) - ord('a')
        answer = options[answer_idx]
        question = row["article"] + "\n\n" + row["question"]
        return pd.Series({
            'question': question,
            'options': options,
            'answer': answer,
            'answer_idx': answer_idx
        })
    df_race = df_race.apply(extract_options_and_answer, axis=1)
    df_race["answer_idx"] = df_race["answer_idx"].astype(int)
    print("  size: ", len(df_race.index))
    return df_race


def load_arc():
    """
    Load the ARC dataset from HuggingFace.
    Returns a pandas DataFrame with standardized columns.
    """
    print("Fetching ARC dataset...")
    df_arc = pd.DataFrame(
        load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            split="train")
    ).sample(frac=1).head(1000).reset_index(drop=True)
    def extract_options_and_answer(row):
        choices = row['choices']
        labels = choices['label']
        texts = choices['text']
        answer_key = row['answerKey']
        # Create a mapping from label to text
        label_to_text = dict(zip(labels, texts))
        # Extract options as a list
        options = texts
        # Get the correct answer text based on answerKey
        answer = label_to_text.get(answer_key, None)
        answer_idx = options.index(answer)
        return pd.Series({
            'question': row['question'],
            'options': options,
            'answer': answer,
            'answer_idx': answer_idx
        })
    df_arc = df_arc.apply(extract_options_and_answer, axis=1)
    df_arc["answer_idx"] = df_arc["answer_idx"].astype(int)
    print("  size: ", len(df_arc.index))
    return df_arc


def load_commonsenseqa():
    """
    Load the CommonsenseQA dataset from HuggingFace.
    Returns a pandas DataFrame with standardized columns.
    """
    print("Fetching CommonsenseQA dataset...")
    df_csqa = pd.DataFrame(
        load_dataset(
            "commonsense_qa",
            split="train"
        )
    ).sample(frac=1).head(1000).reset_index(drop=True)
    def extract_options_and_answer(row):
        choices = row['choices']
        labels = choices['label']
        texts = choices['text']
        answer_key = row['answerKey']
        # Create a mapping from label to text
        label_to_text = dict(zip(labels, texts))
        # Extract options as a list
        options = texts
        # Get the correct answer text based on answerKey
        answer = label_to_text.get(answer_key, None)
        options.remove(answer)
        if len(options) < 3:
            return pd.Series({})
        options = random.sample(options, 3) + [answer]
        answer_idx = options.index(answer)
        return pd.Series({
            'question': row['question'],
            'options': options,
            'answer': answer,
            'answer_idx': answer_idx
        })
    df_csqa = df_csqa.apply(extract_options_and_answer, axis=1)
    df_csqa["answer_idx"] = df_csqa["answer_idx"].astype(int)
    print("  size: ", len(df_csqa.index))
    return df_csqa


def load_triviaqa():
    """
    Load the TriviaQA dataset from HuggingFace.
    Returns a pandas DataFrame with standardized columns.
    """
    print("Fetching TriviaQA dataset...")
    df_trivia = pd.DataFrame(
        load_dataset(
            "trivia_qa",
            "rc",
            split="train",
            streaming=True
        ).take(2000)
    )
    def extract_options_and_answer(row):
        question = row['question']
        answer_dict = row['answer']

        # Extract the correct answer from 'normalized_value'
        answer = answer_dict.get('normalized_value', '').strip().lower()
        options = answer_dict.get('normalized_aliases', [])
        if answer not in options:
            return pd.Series({})
        options.remove(answer)
        if len(options) < 3:
            return pd.Series({})
        options = random.sample(options, 3) + [answer]
        answer_idx = options.index(answer)
        return pd.Series({
            'question': question,
            'options': options,
            'answer': answer,
            'answer_idx': answer_idx
        })
    df_trivia = df_trivia.apply(extract_options_and_answer, axis=1).dropna().head(1000)
    df_trivia["answer_idx"] = df_trivia["answer_idx"].astype(int)
    print("  size: ", len(df_trivia.index))
    return df_trivia


def main():
    # Load datasets
    df_race = load_race()
    df_race.to_csv(OUTPUT_DIR + "/race-sample.csv.gz", index=False, compression="gzip")
    df_race["source"] = "race"

    df_arc = load_arc()
    df_arc.to_csv(OUTPUT_DIR + "/arc-sample.csv.gz", index=False, compression="gzip")
    df_arc["source"] = "arc"

    df_csqa = load_commonsenseqa()
    df_csqa.to_csv(OUTPUT_DIR + "/csqa-sample.csv.gz", index=False, compression="gzip")
    df_csqa["source"] = "csqa"

    df_trivia = load_triviaqa()
    df_csqa.to_csv(OUTPUT_DIR + "/trivia-sample.csv.gz", index=False, compression="gzip")
    df_trivia["source"] = "trivia"

    # Combine datasets
    df = pd.concat([df_race, df_arc, df_csqa, df_trivia], ignore_index=True).sample(frac=1)
    df_over_4_choices = df[df["options"].apply(len) > 4]
    if len(df_over_4_choices) > 0:
        print("WARNING: questions with more than 4 choices were selected. count: ", len(df_over_4_choices))
    df.to_csv(DATA_DIR + "/mcq-all.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
