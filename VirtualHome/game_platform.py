import json
import os.path
import re

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

from perceive import *
from llm.run_gpt_prompt import *

from llm.gpt_request import get_embedding

from actions import actions_description

import math
from numpy import dot
from numpy.linalg import norm


def calculate_distance(coord1, coord2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))


def find_nearest_object(objects, target_coord):
    nearest_object = None
    min_distance = float('inf')

    for obj in objects:
        obj_coord = obj['obj_transform']['position']
        distance = calculate_distance(obj_coord, target_coord)

        if distance < min_distance:
            min_distance = distance
            nearest_object = obj

    return nearest_object


def sort_objects_by_distance(objects, target_coord):
    return sorted(objects, key=lambda obj: calculate_distance(obj['obj_transform']['position'], target_coord))


def fix_item_name(rough_command):
    if 'coffee maker' in rough_command:
        rough_command = rough_command.replace("coffee maker", "coffeemaker")

    if 'coffee_maker' in rough_command:
        rough_command = rough_command.replace("coffee_maker", "coffeemaker")

    if 'living room' in rough_command:
        rough_command = rough_command.replace("living room", "livingroom")

    return rough_command


def get_item_class_set(node_list):
    class_set = []
    for node in node_list:
        if node["class_name"] not in class_set:
            class_set.append(node["class_name"])
    return class_set


def get_item_properties(relative_items_node):
    item_properties = {}
    for k, v in relative_items_node.items():
        if len(v) > 0:
            item_properties[k] = v[0]['properties']
        else:
            assert False, f"get_item_properties -> {k} not found items"
    return item_properties


def get_item_relation_description(agent_item_relation):
    prompt = ''
    # FACING
    if 'FACING' in agent_item_relation.keys() and len(agent_item_relation['FACING']) > 0:
        prompt += f"items you are facing: {','.join(get_item_class_set(agent_item_relation['FACING']))}\n"
    # CLOSE
    if 'CLOSE' in agent_item_relation.keys() and len(agent_item_relation['CLOSE']) > 0:
        prompt += f"items close to you: {','.join(get_item_class_set(agent_item_relation['CLOSE']))}\n"
    else:
        prompt += f"items close to you: nothing\n"
    # HOLDS_RH
    if 'HOLDS_RH' in agent_item_relation.keys() and len(agent_item_relation['HOLDS_RH']) > 0:
        prompt += f"your right hand holding: {','.join(get_item_class_set(agent_item_relation['HOLDS_RH']))}\n"
    # HOLDS_LH
    if 'HOLDS_LH' in agent_item_relation.keys() and len(agent_item_relation['HOLDS_LH']) > 0:
        prompt += f"your left hand holding: {','.join(get_item_class_set(agent_item_relation['HOLDS_LH']))}\n"

    if 'HOLDS_RH' not in agent_item_relation.keys() and 'HOLDS_LH' not in agent_item_relation.keys():
        prompt += f"You are not holding anything in either of your hands.\n"
    # SITTING
    if 'SITTING' in agent_item_relation.keys() and len(agent_item_relation['SITTING']) > 0:
        prompt += f"you are sitting on: {','.join(get_item_class_set(agent_item_relation['SITTING']))}\n"
    return prompt


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


class Platform(object):
    def __init__(self, save_fold_path, args):
        self.args = args
        self.use_policy = args.disable_policy

        self.comm = None
        self.nodes = None
        self.edges = None

        self.daily_plan_list = None
        self.env_graph_list = []

        self.hourly_plan_action_dict = {}
        self.hourly_plan_state_dict = {}

        self.render_step = 0
        self.save_path = os.path.join(save_fold_path, 'frame_graph')
        os.makedirs(self.save_path)

        public_fold = "./data/public"
        if not os.path.isdir(public_fold):
            os.makedirs(public_fold)

        self.putback_forbidden_items = []  # there are some bug in the virtualhome environment
        self.putback_forbidden_items_path = os.path.join(os.path.join(public_fold, "putback_forbidden_item.json"))
        if os.path.isfile(self.putback_forbidden_items_path):
            with open(self.putback_forbidden_items_path, 'r') as f:
                self.putback_forbidden_items = json.load(f)['items']

        self.policy_fold = os.path.join(public_fold, 'policy')
        if not os.path.isdir(self.policy_fold):
            os.makedirs(self.policy_fold)

        self.policy = {}
        if self.use_policy:
            for file in os.listdir(self.policy_fold):
                if 'json' not in file:
                    continue
                file_path = os.path.join(self.policy_fold, file)
                with open(file_path, 'r') as f:
                    json_file = json.load(f)
                    self.policy[json_file['plan_name']] = json_file

        self.clear_env()

    def run(self):
        self.update_graph()
        self.daily_plan()
        for hourly_plan in self.daily_plan_list:
            print(f"Current Task: {hourly_plan}")
            success = self.hourly_plan_policy(hourly_plan)
            if not success:
                self.hourly_plan(hourly_plan)

    def save(self, save_fold):
        json_data = {}
        json_data['daily_plan'] = self.daily_plan_list
        json_data['hourly_plan'] = {}
        for hourly_plan in self.daily_plan_list:
            json_data['hourly_plan'][hourly_plan] = {}
            json_data['hourly_plan'][hourly_plan]['state'] = self.hourly_plan_state_dict[hourly_plan]
            json_data['hourly_plan'][hourly_plan]['action'] = self.hourly_plan_action_dict[hourly_plan]

        with open(os.path.join(save_fold, "plan_action.json"), 'w') as f:
            json.dump(json_data, f, indent=4)

    def render(self, command, update=True, rough_command=None):
        if not isinstance(command, list):
            command = [command]
        success, message = self.comm.render_script(command, recording=False, skip_animation=True,
                                                   find_solution=True)
        if success and update:
            self.update_graph()

        print(f"command:{command} state:{success} reason:{message}")

        return success, message

    def update_graph(self):
        _, env_graph = self.comm.environment_graph()
        self.nodes = env_graph["nodes"]
        self.edges = env_graph["edges"]
        self.save_frame_graph()

    def save_frame_graph(self):
        _, env_graph = self.comm.environment_graph()
        with open(os.path.join(self.save_path, f"{self.render_step}.json"), 'w') as f:
            json.dump(env_graph, f, indent=4)
        self.render_step += 1

    def clear_env(self):  # There is a bug in VirtualHome where the agent can't drop items, so it has to be reset after
        # finish the task
        if self.comm is not None:
            self.comm.close()
        self.comm = UnityCommunication(file_name=self.args.unity_filename,
                                       port=self.args.port)
        self.comm.reset(0)
        self.comm.add_character('Chars/Female1')
        self.update_graph()

    def daily_plan(self):
        room_interactive_item = get_room_all_interactive_item_str(self.nodes, self.edges)
        daily_plan = run_gpt_daily_plan(room_interactive_item)

        self.daily_plan_list = []
        for activity in daily_plan:
            self.daily_plan_list.append(activity['activity'])
        print(f"daily_plan: \n{', '.join(self.daily_plan_list)}")

    def hourly_plan(self, hourly_plan):
        self.hourly_plan_action_dict[hourly_plan] = []

        previous_action = []
        previous_command = []

        step = 0
        step_include_try = 0

        relative_items = self.get_interact_item(hourly_plan)
        relative_items_node = self.get_relative_item_node(relative_items)
        relative_items_properties = get_item_properties(relative_items_node)

        success = True
        all_condition_action_list = []
        while True:
            message = ""
            rough_command, true_command, room, agent_info = None, None, None, None
            forbidden_action = []
            condition_node = []
            for i in range(5):
                success = True
                # try select action
                rough_command, true_command, room, agent_info = self.actual_action(hourly_plan, previous_action,
                                                                                   relative_items_properties,
                                                                                   forbidden_action)

                # get condition
                condition_node = get_node_from_command(self.nodes, true_command)

                need_walk, walk_target_node = self.check_need_to_walk(true_command)
                if need_walk:
                    success, walk_rough_command, walk_true_command = self.walk_to_item(hourly_plan, walk_target_node)
                    if success:
                        previous_action.append(walk_rough_command)
                        previous_command.append(walk_true_command)

                if success:
                    success, message = self.render(true_command)

                reason = ''
                if not success:
                    reason = message
                    forbidden_action.append(rough_command)
                # record action
                action_dict = {
                    'rough_command': rough_command,
                    'true_command': true_command,
                    'state': {
                        'success': success,
                        'reason': reason
                    }
                }
                step_include_try += 1
                self.hourly_plan_action_dict[hourly_plan].append(action_dict)

                if success:
                    # save condition_action
                    break
            step += 1
            if not success:
                self.hourly_plan_state_dict[hourly_plan] = {
                    'type': 'fail',
                    'reason': json.dumps(message)
                }
                break
            else:
                previous_action.append(rough_command)
                previous_command.append(true_command)
                task_state = run_gpt_check_task_state(hourly_plan, room, agent_info, previous_action)

                all_condition_action_list.append(
                    {
                        'condition_nodes': condition_node,
                        'action': rough_command
                    }
                )
                if task_state['type'] == 'success':
                    self.hourly_plan_state_dict[hourly_plan] = {
                        'type': task_state['type'],
                        'reason': task_state['reason']
                    }

                    if self.use_policy:
                        self.save_policy(hourly_plan, all_condition_action_list)

                    break
                if step_include_try > 20:
                    print(f"Exceeded maximum number of steps")
                    self.hourly_plan_state_dict[hourly_plan] = {
                        'type': 'fail',
                        'reason': "Exceeded maximum number of steps"
                    }
                    break  # Avoid infinite loops

        self.clear_env()
        # self.put_down_after_finish_task(hourly_plan)

    def actual_action(self, hourly_plan, previous_action, relative_items_properties, forbidden_action):
        current_room = get_current_room(self.nodes, self.edges)  # location

        agent_item_relation = get_agent_info(self.nodes, self.edges)
        agent_info = get_item_relation_description(agent_item_relation)  # agent information

        # candidate actions
        # candidate_actions = \
        #     run_gpt_get_relative_actions(hourly_plan, current_room, agent_info, relative_items_properties_prompt)[
        #         'actions']
        # candidate_actions = list(actions_description.keys())
        candidate_actions = get_legal_action(self.nodes, self.edges)
        # rough next action commands
        if len(forbidden_action) == 0:
            rough_command = run_gpt_get_next_rough_actions(hourly_plan, current_room, agent_info,
                                                           relative_items_properties, set(candidate_actions),
                                                           previous_action)['command']
        else:
            rough_command = run_gpt_get_next_rough_actions_with_forbidden_action(hourly_plan, current_room, agent_info,
                                                                                 relative_items_properties,
                                                                                 set(candidate_actions),
                                                                                 previous_action, forbidden_action)[
                'command']

        rough_command = fix_item_name(rough_command)
        # select id
        split_command = rough_command.split(' ')
        id_list = []
        if len(split_command) > 1:
            for item_name in split_command[1:]:
                candidate_item_description = ""
                item_node_edge_room = get_class_name_node_edge_room(self.nodes, self.edges, item_name)
                for item_id, v in item_node_edge_room.items():
                    node = v['node']
                    edge = v['edge']
                    room = v['room']
                    edge_description = get_edge_description(self.nodes, node, edge)
                    candidate_item_description += f"{item_name}: id({item_id}) location({room}) " \
                                                  f"state({', '.join(node['states'])}) relation_with_other_item({edge_description})\n"
                target_id = run_gpt_get_item_id(hourly_plan, current_room, agent_info, previous_action,
                                                rough_command, item_name, candidate_item_description)['id']
                id_list.append(target_id)

        # change to true command
        true_command = '<char0> '
        true_command += f"[{split_command[0]}] "
        if len(split_command) > 1:
            true_command += f"<{split_command[1]}> ({id_list[0]})"

        if len(split_command) > 2:
            true_command += f" <{split_command[2]}> ({id_list[1]})"

        return rough_command, true_command, current_room, agent_info

    def get_interact_item(self, hourly_plan):
        all_interactive_item = get_all_interactive_item(self.nodes)
        all_interactive_item_name = []
        for node in all_interactive_item:
            if node['class_name'] not in all_interactive_item_name:
                all_interactive_item_name.append(node['class_name'])
        relative_items = run_gpt_get_relative_items(hourly_plan, ','.join(all_interactive_item_name))['items']
        return relative_items

    def get_relative_item_node(self, relative_items):
        relative_items_node = {}
        all_interactive_item = get_all_interactive_item(self.nodes)
        for relative_item_name in relative_items:
            for node in all_interactive_item:
                if node['class_name'] == relative_item_name:
                    if node['class_name'] not in relative_items_node.keys():
                        relative_items_node[node['class_name']] = []
                    relative_items_node[node['class_name']].append(node)
        return relative_items_node

    def check_need_to_walk(self, true_command):
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, true_command)
        if match.group(1) not in ['sit', 'grab', 'open', 'close', 'putback', 'putin', 'switchon', 'switchoff', 'drink',
                                  'touch', 'putback', 'Putback']:
            return False, None

        numbers = re.findall(r'\((\d+)\)', true_command)
        numbers = list(map(int, numbers))
        target_id = numbers[-1]

        id_nodes = get_id_node(self.nodes)

        if target_id not in id_nodes.keys():
            return False, None

        return True, id_nodes[target_id]

    def walk_to_item(self, hourly_plan, node1, record=True):
        rough_command = f'walk {node1["class_name"]}'
        true_command = f'<char0> [walk] <{node1["class_name"]}> ({node1["id"]})'

        success, message = self.render(true_command, update=False)

        if not success:
            print(f"walk_to_item program error {true_command}, {message}")

        print(f"walk_to_item program {true_command}")
        return success, rough_command, true_command

    def put_down_after_finish_task(self, hourly_plan):
        id_node = get_id_node(self.nodes)
        right_hand_with_item = False
        right_hand_node = None
        left_hand_with_item = False
        left_hand_node = None
        agent_node = get_agent_node(self.nodes)
        for edge in self.edges:
            from_id = edge['from_id']
            to_id = edge['to_id']
            relation_type = edge['relation_type']
            if from_id == agent_node['id']:
                if relation_type == "HOLDS_RH" or relation_type == "HOLDS_LH":
                    if relation_type == "HOLDS_RH":
                        right_hand_with_item = True
                        right_hand_node = id_node[to_id]
                    else:
                        left_hand_with_item = True
                        left_hand_node = id_node[to_id]

        if not right_hand_with_item and not left_hand_with_item:
            return

        all_interactive_items = get_all_interactive_item(self.nodes)
        all_surface_items = []
        for item in all_interactive_items:
            if right_hand_with_item and item['id'] == right_hand_node['id']:
                continue
            if left_hand_with_item and item['id'] == left_hand_node['id']:
                continue
            if 'SURFACES' in item['properties']:
                all_surface_items.append(item)

        close_surface_node_list = sort_objects_by_distance(all_surface_items,
                                                           agent_node['obj_transform']['position'])
        for surface_node in close_surface_node_list:
            success, rough_command, true_command = self.walk_to_item(hourly_plan, surface_node)
            success = False
            if right_hand_with_item:
                success, _, _ = self.put_down_item(hourly_plan, right_hand_node, surface_node)
                if success:
                    right_hand_with_item = False
            if left_hand_with_item:
                success, _, _ = self.put_down_item(hourly_plan, left_hand_node, surface_node)
                if success:
                    left_hand_with_item = False
            if not right_hand_with_item and not left_hand_with_item:
                return

    def put_down_item(self, hourly_plan, node1, node2):
        rough_command = f'putback {node1["class_name"]} {node2["class_name"]}'
        true_command = f'<char0> [PutBack] <{node1["class_name"]}> ({node1["id"]}) <{node2["class_name"]}> ({node2["id"]})'

        success, message = self.render(true_command)

        action_dict = {
            'rough_command': rough_command,
            'true_command': true_command,
            'state': {
                'success': success,
                'reason': "generated by program, " + json.dumps(message)
            }
        }
        if not success:
            print(f"put_down_item program error <{true_command}>: {json.dumps(message)}")
            # assert False, f"put_down_item program error {true_command}, {message}"
        else:
            print(f"put_down_item program {true_command}")
            self.hourly_plan_action_dict[hourly_plan].append(action_dict)
        return success, rough_command, true_command

    def hourly_plan_policy(self, hourly_plan):
        if not self.use_policy:
            return False

        target_plan = self.get_related_plan(hourly_plan)
        if target_plan is None:
            if self.args.evaluation_mode:
                self.hourly_plan_state_dict[hourly_plan] = {
                    'type': 'fail',
                    'reason': 'fail to generate'
                }
                self.hourly_plan_action_dict[hourly_plan] = {}
                return True

            return False

        for candidate_action_list in self.policy[target_plan]['condition_actions']:
            success, command_list = self.compare_condition(candidate_action_list)
            if success:
                self.actual_action_policy(hourly_plan, command_list)
                self.clear_env()
                return True

        return False

    def get_related_plan(self, hourly_plan):
        if hourly_plan in self.policy.keys():
            print(f'hourly_plan_policy -> find target plan ({hourly_plan})')
            return hourly_plan
        else:
            embedding = get_embedding(hourly_plan)
            for plan_name, info in self.policy.items():
                score = cos_sim(embedding, info['embedding'])
                if score >= 0.97:
                    print(f'hourly_plan_policy -> find related plan ({hourly_plan}) -> ({plan_name})')
                    return plan_name
        return None

    def compare_condition(self, condition_action_pair_list):
        id_node = get_id_node(self.nodes)

        true_action_command_list = []
        for condition_action_pair in condition_action_pair_list:
            condition = condition_action_pair['condition']
            rough_action = condition_action_pair['action']
            action_id_list = []
            for item in condition:
                class_name = item['class_name']
                states = item['states']
                item_id = item['id']
                if item_id in id_node.keys():
                    target_item = id_node[item_id]
                    if target_item['class_name'] == class_name and target_item['states'] == states:
                        action_id_list.append(item_id)
                        continue
                similar_node_list = get_same_state_class_nodes(self.nodes, states, class_name)
                if len(similar_node_list) == 0:
                    return False, None
                action_id_list.append(similar_node_list[0])
            split_action = rough_action.split(" ")
            true_action_command = f"<char0> [{split_action[0]}]"
            for item_name, item_id in zip(split_action[1:], action_id_list):
                true_action_command += f' <{item_name}> ({item_id})'
            true_action_command_list.append(true_action_command)

        return True, true_action_command_list

    def actual_action_policy(self, hourly_plan, command_list):
        all_command = []
        self.hourly_plan_action_dict[hourly_plan] = []
        for command in command_list:
            self.check_need_to_walk(command)
            need_walk, walk_target_node = self.check_need_to_walk(command)
            if need_walk:
                true_command = f'<char0> [walk] <{walk_target_node["class_name"]}> ({walk_target_node["id"]})'
                all_command.append(true_command)
            all_command.append(command)
            action_dict = {
                'rough_command': '',
                'true_command': command,
                'state': {
                    'success': True,
                    'reason': 'use existed policy'
                }
            }
            self.hourly_plan_action_dict[hourly_plan].append(action_dict)
        self.hourly_plan_state_dict[hourly_plan] = {
            'type': 'success',
            'reason': 'use existed policy'
        }
        success, message = self.render(all_command)
        if not success:
            assert False, 'policy fail'

    def save_policy(self, hourly_plan, condition_nodes_action_list):
        json_data = {}
        target_plan = hourly_plan
        related_plan = self.get_related_plan(hourly_plan)
        if related_plan is None:
            json_data['plan_name'] = hourly_plan
            json_data['embedding'] = get_embedding(hourly_plan)
            json_data['file_name'] = f"{len(self.policy)}.json"
            json_data['condition_actions'] = []
        else:
            json_data = self.policy[related_plan]
            target_plan = related_plan


        candidate_plan = []
        for condition_action in condition_nodes_action_list:
            condition_nodes = condition_action['condition_nodes']
            command = condition_action['action']
            condition = []
            for node in condition_nodes:
                condition.append(
                    {
                        "class_name": node['class_name'],
                        "states": node['states'],
                        "id": node['id']
                    }
                )
            candidate_plan.append(
                {
                    "condition": condition,
                    "action": command
                }
            )
        json_data['condition_actions'].append(candidate_plan)
        self.policy[target_plan] = json_data

        with open(os.path.join(self.policy_fold, json_data['file_name']), 'w') as f:
            json.dump(json_data, f, indent=4)



