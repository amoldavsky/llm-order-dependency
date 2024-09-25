from _dirs import *
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


# Function to analyze correctness by position and certainty
def calculate_correctness(df, pos):
    # Calculate per position correctness
    df[f'pos_{pos}_correctness'] = (df[f'pos_{pos}_response_idx'] == pos).astype(int)

    correctness_col = f'pos_{pos}_correctness'

    # Overall correctness at this position
    overall_correctness = df[correctness_col].mean()

    # Correctness for certain questions
    df_certain = df[df["is_certain"]]
    certain_correctness = df_certain[correctness_col].mean()

    # Correctness for uncertain questions
    df_uncertain = df[~df["is_certain"]]
    uncertain_correctness = df_uncertain[correctness_col].mean()

    return overall_correctness, certain_correctness, uncertain_correctness


def calculate_confusion_matrix(df, positions):
    # Prepare actual and predicted positions
    actual_positions = []
    predicted_positions = []

    for pos in positions:
        actual_pos = [pos] * len(df)
        actual_positions.extend(actual_pos)
        predicted_pos = df[f'pos_{pos}_response_idx'].tolist()
        predicted_positions.extend(predicted_pos)

    # Create confusion matrix
    cm = confusion_matrix(actual_positions, predicted_positions, labels=positions)

    df_cm = pd.DataFrame(
        cm, index=[f"Actual {pos}" for pos in positions],
        columns=[f"Predicted {pos}" for pos in positions])

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm_norm = pd.DataFrame(
        cm_norm, index=[f"Actual {pos}" for pos in positions],
        columns=[f"Predicted {pos}" for pos in positions])
    return df_cm, df_cm_norm


def calculate_additional_metrics(df, pos):
    actual = [pos] * len(df)
    predicted = df[f'pos_{pos}_response_idx'].tolist()

    # Binary classification
    actual_binary = [1 if a == pos else 0 for a in actual]
    predicted_binary = [1 if p == pos else 0 for p in predicted]

    precision = precision_score(actual_binary, predicted_binary, zero_division=0)
    recall = recall_score(actual_binary, predicted_binary, zero_division=0)
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0)

    return precision, recall, f1


def calculate_roc_auc(df, pos):
    # TODO
    return None


def calculate_response_distribution(df, pos):
    """
    Calculates the percentage distribution of the model's choice/response
    for a specific locked position test run.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - pos (int): The locked position (0-3) for which to calculate the distribution.

    Returns:
    - pd.Series: A series representing the percentage distribution of predictions.
    """

    # Calculate value counts normalized to get percentages, ensuring all positions are represented
    distribution_percentage = \
            df[f'pos_{pos}_response_idx']\
                .value_counts(normalize=True)\
                .reindex([0, 1, 2, 3], fill_value=0) * 100

    # Rename the index for clarity
    distribution_percentage.index = [f"options {p}" for p in distribution_percentage.index]

    # Round the percentages to two decimal places
    return distribution_percentage.round(2)


def analyze_dataset(df, name):
    print("analyzing dataset ", name)
    print("  question count: ", len(df.index))

    # Analyze correctness by position
    positions = [0, 1, 2, 3]
    overall_correctness_list = []
    certain_correctness_list = []
    uncertain_correctness_list = []

    for pos in positions:
        print("  results position ", pos)
        print("    correctness:")
        overall_corr, certain_corr, uncertain_corr = calculate_correctness(df, pos)
        print(f"      overall: {overall_corr:.4f}")
        print(f"      across certain questions: {certain_corr:.4f}")
        print(f"      across uncertain questions: {uncertain_corr:.4f}")
        overall_correctness_list.append(overall_corr)
        certain_correctness_list.append(certain_corr)
        uncertain_correctness_list.append(uncertain_corr)

        # distribution of choices
        df_resp_ds_certain = calculate_response_distribution(df[df["is_certain"]], pos)
        print("    choice distribution (certain questions): ")
        print("      options 0: ", df_resp_ds_certain.iloc[0])
        print("      options 1: ", df_resp_ds_certain.iloc[1])
        print("      options 2: ", df_resp_ds_certain.iloc[2])
        print("      options 3: ", df_resp_ds_certain.iloc[3])
        df_resp_ds_uncertain = calculate_response_distribution(df[~df["is_certain"]], pos)
        print("    choice distribution (uncertain question): ")
        print("      options 0: ", df_resp_ds_uncertain.iloc[0])
        print("      options 1: ", df_resp_ds_uncertain.iloc[1])
        print("      options 2: ", df_resp_ds_uncertain.iloc[2])
        print("      options 3: ", df_resp_ds_uncertain.iloc[3])

        # metrics: Precision, Recall, F1-Score
        precision, recall, f1 = calculate_additional_metrics(df, pos)
        print("    precision: ", precision)
        print("    recall: ", recall)
        print("    f1: ", f1)
        # # metrics: ROC AUC
        # roc_auc = calculate_roc_auc(df, pos)
        # print("    ROC AUC: ", roc_auc)

    # confusion matrix
    (df_cm_certain, df_cm_certain_norm) = calculate_confusion_matrix(
        df[df["is_certain"]], positions)
    print("\n  confusion matrix for certain questions (counts):")
    print(df_cm_certain)
    print("\n  normalized confusion matrix for certain questions (percentage):")
    print(df_cm_certain_norm.round(2))
    (df_cm_uncertain, df_cm_certain_norm) = calculate_confusion_matrix(
        df[~df["is_certain"]], positions)
    print("\n  confusion matrix for uncertain questions (counts):")
    print(df_cm_certain)
    print("\n  normalized confusion matrix for uncertain questions (percentage):")
    print(df_cm_certain_norm.round(2))

    # Statistical Testing: Chi-Square Test of Independence
    # Prepare the contingency table
    contingency_table = np.zeros((4, 2))  # 4 positions x 2 outcomes (Correct, Incorrect)

    for idx, pos in enumerate(positions):
        correctness_col = f'pos_{pos}_correctness'
        correct = df[df[correctness_col] == 1].shape[0]
        incorrect = df[df[correctness_col] == 0].shape[0]
        contingency_table[idx, 0] = correct
        contingency_table[idx, 1] = incorrect

    # Perform Chi-Square Test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print("  chi-Square test:")
    print(f"    chi-Square Statistic: {chi2:.4f}")
    print(f"    Degrees of Freedom: {dof}")
    print(f"    p-value: {p_value:.4e}")

    if p_value < 0.05:
        print("    Result: The difference in correctness across positions is statistically significant (p < 0.05).")
    else:
        print("    Result: The difference in correctness across positions is not statistically significant (p >= 0.05).")


if __name__ == "__main__":
    # Load the data
    print("Loading GPT-3.5 results...")
    df_gpt35 = pd.read_csv(DIST_DIR + '/mcq-gpt-35-turbo_test.csv')
    analyze_dataset(df_gpt35, "gpt-3.5-turbo")
    print("\n")

    print("Loading GPT-4o-mini results...")
    df_gpt4om = pd.read_csv(DIST_DIR + '/mcq-gpt-4o-mini_test.csv')
    analyze_dataset(df_gpt4om, "gpt-4o-mini")

