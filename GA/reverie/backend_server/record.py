import os

import utils
from utils import *
import atexit
import json

record_tree = dict()


def update_record_tree(agent_name, task, parent, is_new, duration, time, chat):
    if record_tree_flag:
        if agent_name not in record_tree.keys():
            record_tree[agent_name] = []
        record_tree[agent_name].append({
            'task': task,
            "parent": parent,
            "is_new": is_new,
            "duration": duration,
            "time": str(time),
            "chat": chat
        })


def save_record_tree():
    if record_tree_flag:
        save_path = os.path.join(utils.sim_fold, "metrics")
        for name, info in record_tree.items():
            with open(os.path.join(save_path, f"{name}.json"), 'w') as f:
                json.dump(info, f, indent=4)
                print(f"write record_tree to {os.path.join(save_path, f'{name}.json')}")


atexit.register(save_record_tree)
