import ast
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langchain_openai.chat_models import (
    ChatOpenAI
)
import time
import random
from src.test_prompt import (
    question, instructions
)
import asyncio
import json
import dotenv


dotenv.load_dotenv()


# Custom callback function to be called on retry
def on_retry(retry_state):
    print(f"Error: {retry_state.outcome.exception()}. Retrying... Attempt: {retry_state.attempt_number}")

# Retry configuration: Exponential backoff with max 5 attempts
@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(openai.RateLimitError),
       after=on_retry)  # Call on_retry before each retry
def process_row_with_backoff(row, chat, process_row):
    return process_row(row, chat)


# Asynchronous batch scoring a dataset
async def batch_score(chat, df, process_row, max_workers=10):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, process_row_with_backoff, row, chat, process_row)
            for _, row in df.iterrows()
        ]
        responses = await asyncio.gather(*tasks)
    return responses


def relocate_answer(row, idx=0, do_randomize=True):
    """ Relocates answer to a given index within options / choices """
    options = eval(row["options"])
    answer = row["answer"]
    options.remove(answer)
    if do_randomize:
        random.shuffle(options)
    options.insert(idx, answer)
    row["options"] = options
    row["answer_idx"] = idx
    return row


def score_row(row: dict, chat: ChatOpenAI):
    """ Helper function to score row against an OpenAI model """
    # random delay to avoid rate limits
    time.sleep(random.uniform(50, 500) / 1000)

    prompt = question.format(row["question"], str(row["options"])).strip()
    response_raw = chat.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=prompt)
    ])
    try:
        # response_idx = int(response_raw.content)
        # return response_idx
        response = parse_json(response_raw.content.strip())[0]
        # attempt fix broken response
        response_idx_str = f"{response["response_idx"]}"
        if len(response_idx_str) > 1 or ord(response_idx_str) > ord('3'):
            print("warning: broken response: ", response)
            print(row)
            response["response_idx"] = row["options"].index(response["response_idx"])
        return response
    except Exception as e:
        # an error, we really don't care
        return None


def score_dataset(
        df,
        model="gpt-3.5-turbo",
        times=3,
        process_row=score_row,
        randomize_options=False
) -> []:
    """ Scores a dataset against a given model a number of times."""
    chat = ChatOpenAI(
        model_name=model,
        temperature=0.5,
        top_p=1.0
    )
    dfs = []
    for i in range(times):
        print("scoring run ", i)
        df_run = df.copy()
        # randomize options is needed
        if randomize_options:
            def randomize_options(row):
                random.shuffle(row["options"])
                row["answer_idx"] = row["options"].index(row["answer"])
                return row
            df_run = df_run.apply(randomize_options, axis=1)
        responses = asyncio.run(batch_score(chat, df_run, process_row))
        # df_run["llm_answer_idx"] = responses
        if responses is not None:
            df_run["response_json"] = responses
            df_run["response_idx"] = df_run["response_json"].apply(lambda x: x["response_idx"])
            df_run["response_idx"] = df_run["response_idx"].astype(int) # can break
            df_run["response_proba"] = df_run["response_json"].apply(lambda x: x["proba"])
            df_run["response"] = df_run.apply(lambda row: row["options"][row["response_idx"]], axis=1)
        else:
            df_run["response_json"] = None
            df_run["response_idx"] = -1
            df_run["response_proba"] = None
            df_run["response"] = None
        dfs.append(df_run)
        print("  correctness: ", len(df_run[df_run["response_idx"] == df_run["answer_idx"]].index) / len(df.index))
    return dfs


def parse_json_obj(response: str) -> (dict, str):
    si = response.find("{")
    si = si if si > -1 else 0
    ei = response.rfind("}", si)
    data_str = response[si:ei + 1]
    data_json = json.loads(data_str)
    data_json_str = json.dumps(data_json)
    return data_json, data_json_str


def parse_json_arr(response: str) -> (dict, str):
    si = response.find("[")
    si = si if si > -1 else 0
    ei = response.rfind("]", si)
    data_str = response[si:ei + 1]
    data_json = json.loads(data_str)
    data_json_str = json.dumps(data_json)
    return data_json, data_json_str


def parse_json(response: str) -> (dict, str):
    si1 = response.find("[")
    if si1 < 0:
        return parse_json_obj(response)

    si2 = response.find("{")
    if si2 < 0:
        return parse_json_arr(response)

    if si1 < si2:
        return parse_json_arr(response)
    else:
        return parse_json_obj(response)