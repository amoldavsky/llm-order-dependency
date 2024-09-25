instructions = """
Your job is to process multiple choice questions to the best of your knowledge.
You will be given a question under QUESTION, you will be given 4 answer choices under ANSWERS as array.

** OBJECTIVE **
- Respond with json schema RESPONSE_SCHEMA only
    - response_idx: integer 0, 1, 2, or 3, from ANSWERS only.
    - proba: probability of you picking the answer at every index from ANSWERS.

RESPONSE_SCHEMA:
{
    "proba": ["probability of answer index"],
    "response_idx": "your answer as integer 0, 1, 2, or 3, from ANSWERS, no text before or after the answer!"
}
Respond with JSON only!  
""".strip()

# instructions = """
# Your job is to process multiple choice questions to the best of your knowledge.
# You will be given a question under QUESTION, you will be given 4 answer choices under ANSWERS as array.
#
# ** OBJECTIVE **
# - Respond with a zero-based index of the answer from ANSWERS only.
# - Respond only with number, no text before or after the answer.
#
# Respond only with an integer, no text!
# """.strip()

question = """
QUESTION:
{}

ANSWERS:
{}
""".strip()