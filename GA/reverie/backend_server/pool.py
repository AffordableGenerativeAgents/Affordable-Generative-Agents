import json
import atexit
import os

from utils import *
import openai
from numpy import dot
from numpy.linalg import norm


def get_embedding(text):
    model = "text-embedding-ada-002"
    response = openai.Embedding.create(
        api_base=openai_api_base,
        api_key=openai_api_key,
        api_type=openai_api_type,
        api_version=openai_api_version,
        input=[text],
        engine=model)['data'][0]['embedding']
    return response


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


# embedding
embedding_pool = dict()
if use_embedding_pool and os.path.isfile(embedding_pool_path):
    with open(embedding_pool_path, 'r') as _f:
        embedding_pool = json.load(_f)
        # print(f"embedding_pool -> {embedding_pool}")


def get_embedding_pool(text):
    if use_embedding_pool and text in embedding_pool.keys():
        print(f"embedding_pool -> {text} is in embedding_pool")
        return embedding_pool[text]

    return None


def update_embedding_pool(text, embedding):
    if use_embedding_pool:
        embedding_pool[text] = embedding


def save_embedding_pool():
    if use_embedding_pool:
        with open(embedding_pool_path, 'w') as f:
            json.dump(embedding_pool, f, indent=4)
            print(f"write embedding_pool to {embedding_pool_path}")


# policy
policy_pool = dict()


def load_policy_pool():
    if use_policy_pool:
        agent_list = os.listdir(policy_pool_path)
        for agent_file in agent_list:
            agent_name = agent_file.split(".")[0]
            agent_file_path = os.path.join(policy_pool_path, agent_file)
            with open(agent_file_path, 'r') as f:
                policy_pool[agent_name] = json.load(f)


load_policy_pool()


def get_policy_pool(agent_name, task, duration):
    if use_policy_pool:
        if agent_name not in policy_pool.keys():
            print(f"get_policy_pool -> {agent_name}:{task}:{duration} not in policy_pool")
            policy_pool[agent_name] = dict()
        if task in policy_pool[agent_name].keys():
            if policy_pool[agent_name][task]["duration"] == duration:
                print(f"get_policy_pool -> {agent_name}:{task}:{duration} find")
                return policy_pool[agent_name][task]["decomp"]
            else:
                print(f"get_policy_pool -> {agent_name}:{task}:{duration} wrong duration, "
                      f"should be {policy_pool[agent_name][task]['duration']}")
    return None


def get_relate_policy(agent_name, _task, duration, embedding):
    if use_policy_pool:
        if agent_name not in policy_pool.keys():
            print(f"get_relate_policy -> {agent_name}:{_task}:{duration} not in policy_pool")
            policy_pool[agent_name] = dict()

        for task, info in policy_pool[agent_name].items():
            if task == _task:
                continue
            score = cos_sim(embedding, info["embedding"])
            if score > 0.97:
                print(f"get_relate_policy -> {agent_name}:{task}:{duration} wrong duration, "
                      f"should be {policy_pool[agent_name][task]['duration']}")
                return info['decomp'], task
    return None, None


def update_policy_pool(agent_name, task, duration, policy, embedding):
    if use_policy_pool:
        if task in policy_pool[agent_name].keys():
            count = 1
            while True:
                if f"{task}_{count}" not in task in policy_pool[agent_name].keys():
                    task = f"{task}_{count}"
                    break
                count += 1
        policy_pool[agent_name][task] = dict()
        policy_pool[agent_name][task]["duration"] = duration
        policy_pool[agent_name][task]["decomp"] = policy
        policy_pool[agent_name][task]["embedding"] = embedding


def save_policy_pool():
    if use_embedding_pool:
        for agent_name, info in policy_pool.items():
            file_path = os.path.join(policy_pool_path, f"{agent_name}.json")
            with open(file_path, 'w') as f:
                json.dump(info, f, indent=4)
                print(f"write policy_pool to {file_path}")


# sub_task
sub_task_pool = dict()


def load_sub_task_pool():
    if use_sub_task_pool:
        agent_list = os.listdir(sub_task_pool_path)
        for agent_file in agent_list:
            agent_name = agent_file.split(".")[0]
            agent_file_path = os.path.join(sub_task_pool_path, agent_file)
            with open(agent_file_path, 'r') as f:
                sub_task_pool[agent_name] = json.load(f)


load_sub_task_pool()


def get_sub_task_pool(agent_name, task):
    if use_sub_task_pool:
        if agent_name not in sub_task_pool.keys():
            print(f"get_sub_task_pool -> {agent_name}:{task} not in sub_task_pool")
            sub_task_pool[agent_name] = dict()
        if task in sub_task_pool[agent_name].keys():
            print(f"get_sub_task_pool -> {agent_name}:{task} find")
            return sub_task_pool[agent_name][task]
    print(f"get_sub_task_pool -> {agent_name}:{task} not find")
    return None


def update_sub_task_pool(agent_name, task, info):
    if use_sub_task_pool:
        sub_task_pool[agent_name][task] = info


def save_sub_task_pool():
    if use_sub_task_pool:
        for agent_name, info in sub_task_pool.items():
            file_path = os.path.join(sub_task_pool_path, f"{agent_name}.json")
            with open(file_path, 'w') as f:
                json.dump(info, f, indent=4)
                print(f"write sub_task_pool to {file_path}")


atexit.register(save_embedding_pool)
atexit.register(save_policy_pool)
atexit.register(save_sub_task_pool)
