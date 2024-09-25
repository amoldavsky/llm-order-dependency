# LLM order dependency

Test LLM order dependency against GPT-3.5 and GPT-4o-mini using public datasets. 

## Experiment

To show order dependency with LLMs can: 
(1) pick a few sample MCQs and run them through and LLM many times while shifting answer location
(2) take a large sample of MCQs from various areas and test through the LLM

This experiment focuses on (2) where we sample MCQs from the following datasets:
- RACE
- ARC
- Common Sense QA
- Trivia QA

This experiment also focuses only on 4 multiple choice questions.
This experiment can be extended to test less or more than 4 options.

According to the following papers:
https://arxiv.org/abs/2406.06581
https://arxiv.org/abs/2308.11483

LLMs are more prone to the order dependency problem 
with uncertain questions and are more biased towards the first and last options.  
Therefore, a big chunk of the data preprocessing stage involves
scoring certain vs uncertain questions.

Once we determine certainty for each of the LLMs (gpt-3.5 and gpt-4o-mini) we proceed to
test the actual order dependency. We test answer position at each of the four option slots and record
the LLMs responses. We also record the probabilities for each answer for deeper analysis later.

### Limitations

- Tests only 4 options, we can further experiment with a varying number of multiple choice questions - 2 to 10.
- Tests only 250 rows for order change (openai limits), can further benefit for >1000 rows.
- Does a single run per option shift (openai limits), can further benefit from many runs.
- Was not able to utilize the probabilities the model assigned to each option for each run, can yield interesting insight into the model's choice. 

### Outcomes & Learnings

- Both GPT-3.5-Turbo and GPT-4o-mini exhibit sensitivity to options order change in MCQs.
- Uncertain questions had significantly more fluctuation than certain questions.
- Both models still showed higher correctness for the right option in the uncertain set. 
- GPT-4o-mini has better certainty/understanding of MCQ across a wide range of subject than GPT-3.5-Turbo.
- Was not able to demonstrate the model's preference for the first and last options.

Results can be inspected in Analysis Output below and in [test_analyze.ipynb](src%2Ftest_analyze.ipynb).
You can see a clear variation in responses in `choice distribution` and `confusion matrix`.

### Analysis Output

```shell  
Connected to pydev debugger (build 241.18034.82)
Loading GPT-3.5 results...
analyzing dataset  gpt-3.5-turbo
  question count:  250
  results position  0
    correctness:
      overall: 0.7800
      across certain questions: 0.9370
      across uncertain questions: 0.6179
    choice distribution (certain questions): 
      options 0:  93.7
      options 1:  2.36
      options 2:  3.15
      options 3:  0.79
    choice distribution (uncertain question): 
      options 0:  61.79
      options 1:  17.89
      options 2:  14.63
      options 3:  5.69
    precision:  1.0
    recall:  0.78
    f1:  0.8764044943820225
  results position  1
    correctness:
      overall: 0.7440
      across certain questions: 0.9370
      across uncertain questions: 0.5447
    choice distribution (certain questions): 
      options 0:  2.36
      options 1:  93.7
      options 2:  2.36
      options 3:  1.57
    choice distribution (uncertain question): 
      options 0:  11.38
      options 1:  54.47
      options 2:  20.33
      options 3:  13.82
    precision:  1.0
    recall:  0.744
    f1:  0.8532110091743119
  results position  2
    correctness:
      overall: 0.7240
      across certain questions: 0.8976
      across uncertain questions: 0.5447
    choice distribution (certain questions): 
      options 0:  4.72
      options 1:  4.72
      options 2:  89.76
      options 3:  0.79
    choice distribution (uncertain question): 
      options 0:  17.07
      options 1:  18.7
      options 2:  54.47
      options 3:  9.76
    precision:  1.0
    recall:  0.724
    f1:  0.839907192575406
  results position  3
    correctness:
      overall: 0.6000
      across certain questions: 0.8268
      across uncertain questions: 0.3659
    choice distribution (certain questions): 
      options 0:  5.51
      options 1:  0.79
      options 2:  11.02
      options 3:  82.68
    choice distribution (uncertain question): 
      options 0:  21.14
      options 1:  14.63
      options 2:  27.64
      options 3:  36.59
    precision:  1.0
    recall:  0.6
    f1:  0.75

  confusion matrix for certain questions (counts):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0          119            3            4            1
Actual 1            3          119            3            2
Actual 2            6            6          114            1
Actual 3            7            1           14          105

  normalized confusion matrix for certain questions (percentage):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0         0.94         0.02         0.03         0.01
Actual 1         0.02         0.94         0.02         0.02
Actual 2         0.05         0.05         0.90         0.01
Actual 3         0.06         0.01         0.11         0.83

  confusion matrix for uncertain questions (counts):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0          119            3            4            1
Actual 1            3          119            3            2
Actual 2            6            6          114            1
Actual 3            7            1           14          105

  normalized confusion matrix for uncertain questions (percentage):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0         0.62         0.18         0.15         0.06
Actual 1         0.11         0.54         0.20         0.14
Actual 2         0.17         0.19         0.54         0.10
Actual 3         0.21         0.15         0.28         0.37
  chi-Square test:
    chi-Square Statistic: 22.3549
    Degrees of Freedom: 3
    p-value: 5.5029e-05
    Result: The difference in correctness across positions is statistically significant (p < 0.05).


Loading GPT-4o-mini results...
analyzing dataset  gpt-4o-mini
  question count:  250
  results position  0
    correctness:
      overall: 0.8280
      across certain questions: 0.9836
      across uncertain questions: 0.4030
    choice distribution (certain questions): 
      options 0:  98.36
      options 1:  0.55
      options 2:  0.0
      options 3:  1.09
    choice distribution (uncertain question): 
      options 0:  40.3
      options 1:  17.91
      options 2:  17.91
      options 3:  23.88
    precision:  1.0
    recall:  0.828
    f1:  0.9059080962800875
  results position  1
    correctness:
      overall: 0.8680
      across certain questions: 0.9781
      across uncertain questions: 0.5672
    choice distribution (certain questions): 
      options 0:  1.64
      options 1:  97.81
      options 2:  0.0
      options 3:  0.55
    choice distribution (uncertain question): 
      options 0:  13.43
      options 1:  56.72
      options 2:  11.94
      options 3:  17.91
    precision:  1.0
    recall:  0.868
    f1:  0.9293361884368309
  results position  2
    correctness:
      overall: 0.8640
      across certain questions: 0.9727
      across uncertain questions: 0.5672
    choice distribution (certain questions): 
      options 0:  0.55
      options 1:  2.19
      options 2:  97.27
      options 3:  0.0
    choice distribution (uncertain question): 
      options 0:  13.43
      options 1:  17.91
      options 2:  56.72
      options 3:  11.94
    precision:  1.0
    recall:  0.864
    f1:  0.927038626609442
  results position  3
    correctness:
      overall: 0.7800
      across certain questions: 0.9727
      across uncertain questions: 0.2537
    choice distribution (certain questions): 
      options 0:  1.64
      options 1:  0.0
      options 2:  1.09
      options 3:  97.27
    choice distribution (uncertain question): 
      options 0:  20.9
      options 1:  23.88
      options 2:  29.85
      options 3:  25.37
    precision:  1.0
    recall:  0.78
    f1:  0.8764044943820225

  confusion matrix for certain questions (counts):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0          180            1            0            2
Actual 1            3          179            0            1
Actual 2            1            4          178            0
Actual 3            3            0            2          178

  normalized confusion matrix for certain questions (percentage):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0         0.98         0.01         0.00         0.01
Actual 1         0.02         0.98         0.00         0.01
Actual 2         0.01         0.02         0.97         0.00
Actual 3         0.02         0.00         0.01         0.97

  confusion matrix for uncertain questions (counts):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0          180            1            0            2
Actual 1            3          179            0            1
Actual 2            1            4          178            0
Actual 3            3            0            2          178

  normalized confusion matrix for uncertain questions (percentage):
          Predicted 0  Predicted 1  Predicted 2  Predicted 3
Actual 0         0.40         0.18         0.18         0.24
Actual 1         0.13         0.57         0.12         0.18
Actual 2         0.13         0.18         0.57         0.12
Actual 3         0.21         0.24         0.30         0.25
  chi-Square test:
    chi-Square Statistic: 9.0800
    Degrees of Freedom: 3
    p-value: 2.8246e-02
    Result: The difference in correctness across positions is statistically significant (p < 0.05).

Process finished with exit code 0

```


## Project Structure

`/data` - contains all input and needed data / datasets  
`/tmp` - local temp dir, not included with project  
`/dist` - output distributable resources / final outputs  
`/src` - source code  
`/src/data` - data fetching and generation code, only needed to re-generate all datasets

## Setup

clone
```bash
git clone ...
git lfs install
mkdir ./tmp
```

poetry
```shell
poetry lock
poetry install
```

.env setup
```bash
touch .env
echo "OPENAI_API_KEY=" >> ./.env
```

## Running

[test_analyze.py](src/test_analyze.py) - analysis on pre-scored results  
[test_prompt.py](src/test_prompt.py) - contains the prompts
[test_analyze.ipynb](src/test_analyze.ipynb) - jupyter notebook with analysis and visuals.

(options) [test.py](src/test.py) - re-score order dependency again GPT 3.5 and 4o-mini

(options) [data-process.py](src/data-process.py) - re-score certain vs uncertain questions

(optional) [data.py](src/data.py) - re-create datasets  
*NOTE: this will score a 1000 rows x4 against an openai model *


