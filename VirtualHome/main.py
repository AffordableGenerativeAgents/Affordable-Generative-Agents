import json
import os.path
import sys

sys.path.append("<VirtualHome project Path>")
sys.path.append("<VirtualHome project Path>virtualhome/simulation")

import argparse
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from game_platform import Platform
from llm.metrics import metrics
from datetime import datetime


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", '-p', type=str)
    parser.add_argument("--unity-filename", type=str,
                        default="<The path to VirtualHome executable>")
    parser.add_argument("--port", type=str, default="8080")
    parser.add_argument("--display", type=str, default="0")
    parser.add_argument('--disable_policy', "-d", action='store_false', help='Disable the lifestyle policy')
    parser.add_argument('--evaluation_mode', "-e", action='store_true', help='Agent will not generate the failure '
                                                                             'query plan in evaluation mode')
    return parser.parse_args()


def diff_dict(dict1, dict2):
    diff_keys = dict1.keys() ^ dict2.keys()

    shared_keys = dict1.keys() & dict2.keys()
    diff_values = {k: (dict1[k], dict2[k]) for k in shared_keys if dict1[k] != dict2[k]}

    return diff_keys, diff_values


def find_diff_dicts(list1, list2):
    str_list1 = [str(d) for d in list1]
    str_list2 = [str(d) for d in list2]

    diff_list1 = [eval(d) for d in str_list1 if d not in str_list2]

    diff_list2 = [eval(d) for d in str_list2 if d not in str_list1]

    return diff_list1, diff_list2


def test(comm):
    comm.reset(0)
    comm.add_character('Chars/Female1')
    _, env_graph = comm.environment_graph()
    nodes1 = env_graph["nodes"]
    edges1 = env_graph["edges"]
    success, message = comm.render_script(['<char0> [walk] <apple> (438)'], recording=False, skip_animation=True,
                                          find_solution=True)
    print(success, message)

    _, env_graph = comm.environment_graph()
    nodes2 = env_graph["nodes"]
    edges2 = env_graph["edges"]
    diff1, diff2 = find_diff_dicts(edges1, edges2)
    print("first time:")
    print(f'diff1:{diff1}')
    print(f'diff2:{diff2}')

    success, message = comm.render_script(['<char0> [grab] <apple> (438)'], recording=False, skip_animation=True,
                                          find_solution=True)
    print(success, message)
    _, env_graph = comm.environment_graph()
    nodes3 = env_graph["nodes"]
    edges3 = env_graph["edges"]
    diff1, diff2 = find_diff_dicts(edges2, edges3)
    print("first time:")
    print(f'diff1:{diff1}')
    print(f'diff2:{diff2}')
    exit()


def main(args):
    current_time = datetime.now()
    if args.project_name is None:
        save_fold_path = os.path.join(f"./data/{current_time}")
    else:
        save_fold_path = os.path.join(f"./data/{args.project_name}")
    os.makedirs(save_fold_path)
    game = Platform(save_fold_path, args)

    # metrics -> record token

    metrics.set_fold(save_fold_path)
    # f = open(os.path.join(save_fold_path, 'print.log'), 'w')
    # sys.stdout = f

    game.run()
    game.save(save_fold_path)
    metrics.save()
    # f.close()


if __name__ == '__main__':
    args = opt()
    main(args)
