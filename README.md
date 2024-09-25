# LLM order dependency

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

(options) [test.py](src/test.py) - re-score order dependency again GPT 3.5 and 4o-mini

(options) [data-process.py](src/data-process.py) - re-score certain vs uncertain questions

(optional) [data.py](src/data.py) - re-create datasets  
*NOTE: this will score a 1000 rows x4 against an openai model *

### Output

```shell
/Users/amoldavsky/dev/amoldavsky/llm-order-dependency/llm-order-dependency/.venv/bin/python3 /Users/amoldavsky/dev/amoldavsky/llm-order-dependency/llm-order-dependency/src/test_analyze.py 
Loading GPT-3.5 results...
analyzing dataset  gpt-3.5-turbo
  question count:  250
  results position  0
    correctness:
      overall: 0.7800
      across certain questions: 0.9370
      across uncertain questions: 0.6179
    precision:  1.0
    recall:  0.78
    f1:  0.8764044943820225
  results position  1
    correctness:
      overall: 0.7440
      across certain questions: 0.9370
      across uncertain questions: 0.5447
    precision:  1.0
    recall:  0.744
    f1:  0.8532110091743119
  results position  2
    correctness:
      overall: 0.7240
      across certain questions: 0.8976
      across uncertain questions: 0.5447
    precision:  1.0
    recall:  0.724
    f1:  0.839907192575406
  results position  3
    correctness:
      overall: 0.6000
      across certain questions: 0.8268
      across uncertain questions: 0.3659
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
    precision:  1.0
    recall:  0.828
    f1:  0.9059080962800875
  results position  1
    correctness:
      overall: 0.8680
      across certain questions: 0.9781
      across uncertain questions: 0.5672
    precision:  1.0
    recall:  0.868
    f1:  0.9293361884368309
  results position  2
    correctness:
      overall: 0.8640
      across certain questions: 0.9727
      across uncertain questions: 0.5672
    precision:  1.0
    recall:  0.864
    f1:  0.927038626609442
  results position  3
    correctness:
      overall: 0.7800
      across certain questions: 0.9727
      across uncertain questions: 0.2537
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

