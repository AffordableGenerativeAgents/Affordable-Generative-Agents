import openai
import time
from utils import *
from .metrics import metrics

import inspect


def get_caller_function_names():
    stack = inspect.stack()
    caller_names = [frame.function for frame in stack][2:]
    return '.'.join(caller_names)


def temp_sleep(seconds=0.1):
    if seconds <= 0:
        return
    time.sleep(seconds)

DEBUG = False

def ChatGPT_safe_generate_response(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
    if verbose:
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if DEBUG:
                print(f"input_prompt: ")
                print(f"{prompt}")
                print(f"output_response:")
                print(f"{curr_gpt_response}")
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print(f"---- repeat count: {i}")
                print(curr_gpt_response)
                print("~~~~")

        except Exception as e:
            metrics.fail_record(e)
            pass
    print("FAIL SAFE TRIGGERED")
    return fail_safe_response


def ChatGPT_request(prompt):
    """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response.
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of
                   the parameter and the values indicating the parameter
                   values.
  RETURNS:
    a str of GPT-3's response.
  """
    # temp_sleep()
    try:
        return ChatGPT_single_request(prompt, time_sleep_second=0)
    except Exception as e:
        metrics.fail_record(e)
        print("ChatGPT ERROR")
        return "ChatGPT ERROR"


def ChatGPT_single_request(prompt, time_sleep_second=0.1):
    temp_sleep(time_sleep_second)

    start_time = time.time()
    model_name = "gpt-35-turbo"
    # model_name = "gpt-4"

    if key_type == 'azure':
        completion = openai.ChatCompletion.create(
            api_type=openai_api_type,
            api_version=openai_api_version,
            api_base=openai_api_base,
            api_key=openai_api_key,
            engine=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        message = completion["choices"][0]["message"]["content"]
    elif key_type == 'llama':
        model_name = 'llama-7B'
        completion = openai.ChatCompletion.create(
            api_type=openai_api_type,
            api_version=openai_api_version,
            api_base=openai_api_base,
            api_key=openai_api_key,
            engine=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        message = completion["choices"][-1]["message"]["content"]
    else:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        message = completion["choices"][0]["message"]["content"]

    function_name = get_caller_function_names()
    total_token = completion['usage']['total_tokens']
    time_use = time.time() - start_time
    metrics.call_record(function_name, model_name, total_token, time_use)
    return message


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    start_time = time.time()
    if key_type == 'azure':
        response = openai.Embedding.create(
            api_base=openai_api_base,
            api_key=openai_api_key,
            api_type=openai_api_type,
            api_version=openai_api_version,
            input=[text],
            engine=model)
    else:
        response = openai.Embedding.create(
            input=[text], model=model)

    function_name = get_caller_function_names()
    total_token = response['usage']['total_tokens']
    time_use = time.time() - start_time
    metrics.call_record(function_name, model, total_token, time_use)

    return response['data'][0]['embedding']