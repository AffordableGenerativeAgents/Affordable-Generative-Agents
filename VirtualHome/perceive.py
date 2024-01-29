from actions import actions_description
import re


def get_all_room(nodes):
    ids = []
    names = []
    for node in nodes:
        if node['category'] == 'Rooms':
            ids.append(node['id'])
            names.append(node['class_name'])
    return ids, names


def get_room_all_interactive_item_str(nodes, edges):
    ids, names = get_all_room(nodes)

    all_interactive_item = []
    for node in nodes:
        if len(node['properties']) > 0:
            all_interactive_item.append(node)

    room_interactive_item = {}
    edge_dict = {}
    for edge in edges:
        if edge['relation_type'] == "ON" or edge['relation_type'] == "INSIDE":
            edge_dict[edge['from_id']] = edge['to_id']

    for node in all_interactive_item:
        from_id = node['id']
        if from_id not in edge_dict.keys():
            print(node)
        to_id = edge_dict[from_id]
        while to_id not in ids:
            if to_id not in edge_dict.keys():
                assert False, "error"
            to_id = edge_dict[to_id]
        if to_id not in room_interactive_item.keys():
            room_interactive_item[to_id] = []
        room_interactive_item[to_id].append(node)

    room_interactive_item_str = ''
    for k, v in room_interactive_item.items():
        node_name_list = []
        for node in v:
            node_name_list.append(node['class_name'])
        room_interactive_item_str += f"{names[ids.index(k)]}: {','.join(set(node_name_list))}\n"
    return room_interactive_item_str


def get_all_interactive_item(nodes):
    all_interactive_item = []
    for node in nodes:
        if len(node['properties']) > 0:
            all_interactive_item.append(node)

    return all_interactive_item


def get_agent_info(nodes, edges):
    all_interactive_item = {}
    agent_node = None
    id_node_dict = {}
    for node in nodes:
        if len(node['properties']) > 0:
            all_interactive_item[node['id']] = node

        if node['class_name'] == 'character':
            agent_node = node

        id_node_dict[node['id']] = node

    assert agent_node is not None, "get_agent_info -> not find agent node"

    ret = dict()

    for edge in edges:
        if edge['from_id'] == agent_node["id"] and edge['to_id'] in all_interactive_item.keys():
            if edge['relation_type'] not in ret.keys():
                ret[edge['relation_type']] = []

            ret[edge['relation_type']].append(id_node_dict[edge['to_id']])
    return ret


def get_current_room(nodes, edges):
    ids, names = get_all_room(nodes)

    agent_node = None
    for node in nodes:
        if node['class_name'] == 'character':
            agent_node = node

    assert agent_node is not None, "get_current_room -> not find agent node"

    edge_dict = {}
    for edge in edges:
        if edge['relation_type'] == "ON" or edge['relation_type'] == "INSIDE":
            edge_dict[edge['from_id']] = edge['to_id']

    assert agent_node['id'] in edge_dict.keys(), "get_current_room -> edge_dict not find agent node id"

    to_id = agent_node['id']
    while to_id not in ids:
        assert to_id in edge_dict.keys(), f"get_current_room -> edge_dict not find to_id {to_id}"
        to_id = edge_dict[to_id]

    return names[ids.index(to_id)]


def get_id_room(nodes, edges, item_id):
    ids, names = get_all_room(nodes)

    if item_id in ids:
        return names[ids.index(item_id)]

    agent_node = None
    for node in nodes:
        if node['id'] == item_id:
            agent_node = node

    assert agent_node is not None, f"get_id_room -> not find id node {item_id}"

    edge_dict = {}
    for edge in edges:
        if edge['relation_type'] == "ON" or edge['relation_type'] == "INSIDE":
            edge_dict[edge['from_id']] = edge['to_id']

    assert agent_node['id'] in edge_dict.keys(), f"get_id_room -> edge_dict not find agent node id {agent_node}"

    to_id = agent_node['id']
    while to_id not in ids:
        assert to_id in edge_dict.keys(), f"get_id_room -> edge_dict not find to_id {to_id}"
        to_id = edge_dict[to_id]

    return names[ids.index(to_id)]


def get_id_node(nodes):
    id_node_dict = {}
    for node in nodes:
        id_node_dict[node['id']] = node
    return id_node_dict


def get_class_name_node_edge_room(nodes, edges, item_name):
    ret = {}
    for node in nodes:
        if node['class_name'] == item_name:
            ret[node['id']] = {
                'node': node,
                'edge': [],
                'room': get_id_room(nodes, edges, node['id'])
            }
    for edge in edges:
        from_id = edge['from_id']
        to_id = edge['to_id']
        relation_type = edge['relation_type']
        if from_id in ret.keys():
            ret[from_id]['edge'].append(edge)
        elif to_id in ret.keys():
            ret[to_id]['edge'].append(edge)

    return ret


def get_edge_info(nodes, edges, target_id):
    id_node_dict = get_id_node(nodes)

    from_edge_dict = {}
    to_edge_dict = {}

    for edge in edges:
        from_id = edge['from_id']
        to_id = edge['to_id']
        relation_type = edge['relation_type']
        if from_id == target_id:
            if relation_type not in from_edge_dict:
                from_edge_dict[relation_type] = []
            from_edge_dict[relation_type].append(id_node_dict[to_id]['class_name'])
        if to_id == target_id:
            if relation_type not in to_edge_dict:
                to_edge_dict[relation_type] = []
            to_edge_dict[relation_type].append(id_node_dict[from_id]['class_name'])

    return from_edge_dict, to_edge_dict


def get_edge_description(nodes, node, edge):
    from_edge_dict, to_edge_dict = get_edge_info(nodes, edge, node['id'])

    prompt = ""
    if 'FACING' in from_edge_dict.keys() and len(from_edge_dict['FACING']) > 0:
        prompt += f"facing ({','.join(from_edge_dict['FACING'])}), "
    # CLOSE
    if 'CLOSE' in from_edge_dict.keys() and len(from_edge_dict['CLOSE']) > 0:
        prompt += f"close to ({','.join(from_edge_dict['CLOSE'])}), "

    if 'ON' in from_edge_dict.keys() and len(from_edge_dict['ON']) > 0:
        prompt += f"on ({','.join(from_edge_dict['ON'])}), "

    if 'INSIDE' in from_edge_dict.keys() and len(from_edge_dict['INSIDE']) > 0:
        prompt += f"inside ({','.join(from_edge_dict['INSIDE'])}), "

    if 'FACING' in to_edge_dict.keys() and len(to_edge_dict['FACING']) > 0:
        prompt += f"faced by ({','.join(to_edge_dict['FACING'])}), "

    if 'ON' in to_edge_dict.keys() and len(to_edge_dict['ON']) > 0:
        prompt += f"({','.join(to_edge_dict['ON'])}) is on, "

    if 'INSIDE' in to_edge_dict.keys() and len(to_edge_dict['INSIDE']) > 0:
        prompt += f"({','.join(to_edge_dict['INSIDE'])}) is inside, "

    # HOLDS_RH
    if 'HOLDS_RH' in to_edge_dict.keys():
        prompt += f"is in character's right hand, "

    if 'HOLDS_LH' in to_edge_dict.keys():
        prompt += f"is in character's left hand, "

    if 'SITTING' in to_edge_dict.keys():
        prompt += f"is sited by character, "

    return prompt


def get_legal_action(nodes, edges):
    all_action = list(actions_description.keys())

    id_nodes = get_id_node(nodes)

    all_interactive_item = {}
    agent_node = None
    for node in nodes:
        if len(node['properties']) > 0:
            all_interactive_item[node['id']] = node

        if node['class_name'] == 'character':
            agent_node = node

    assert agent_node is not None, "get_agent_info -> not find agent node"

    can_sit = False
    can_grab = False
    can_open = False
    can_close = False
    right_hand_with_item = False
    left_hand_with_item = False
    container_near = False
    surface_near = False
    can_switch_on = False
    can_switch_off = False
    can_drink = False
    for edge in edges:
        from_id = edge['from_id']
        to_id = edge['to_id']
        relation_type = edge['relation_type']
        if from_id == agent_node['id'] or to_id == agent_node['id']:
            if relation_type == "HOLDS_RH" or relation_type == "HOLDS_LH":
                if relation_type == "HOLDS_RH":
                    right_hand_with_item = True
                else:
                    left_hand_with_item = True
                if "RECIPIENT" in id_nodes[to_id]['properties']:
                    can_drink = True

            if relation_type == "CLOSE":
                target_id = from_id
                if to_id != agent_node['id']:
                    target_id = to_id
                if 'SITTABLE' in id_nodes[target_id]['properties']:
                    can_sit = True
                if 'GRABBABLE' in id_nodes[target_id]['properties']:
                    can_grab = True
                if "CAN_OPEN" in id_nodes[target_id]['properties']:
                    if 'CLOSED' in id_nodes[target_id]['states']:
                        can_open = True
                    else:
                        can_close = True
                if "CONTAINERS" in id_nodes[target_id]['properties']:
                    container_near = True
                if "SURFACE" in id_nodes[target_id]['properties']:
                    surface_near = True
                if "HAS_SWITCH" in id_nodes[target_id]['properties']:
                    if 'OFF' in id_nodes[target_id]['states']:
                        can_switch_on = True
                    else:
                        can_switch_off = True

    if right_hand_with_item and left_hand_with_item:
        can_grab = False

    can_put = False
    can_putin = False
    if right_hand_with_item or left_hand_with_item:
        if container_near:
            can_putin = True
        if surface_near:
            can_put = True

    # Walk
    # Sit
    if not can_sit:
        all_action.remove('Sit')
    # StandUp
    if 'SITTING' not in agent_node['states']:
        all_action.remove('StandUp')
    # Grab
    # if not can_grab:
    #     all_action.remove('Grab')
    # Open
    if not can_open:
        all_action.remove("Open")
    # Close
    if not can_close:
        all_action.remove("Close")
    # Put
    if not can_put:
        all_action.remove("Put")
    # PutIn
    if not can_putin:
        all_action.remove("PutIn")
    # SwitchOn
    # if not can_switch_on:
    #     all_action.remove("SwitchOn")
    # SwitchOff
    # if not can_switch_off:
    #     all_action.remove("SwitchOff")
    # Drink
    if not can_drink:
        all_action.remove("Drink")
    # Touch
    if True:  # TODO
        all_action.remove("Touch")
    # LookAt
    if True:  # TODO
        all_action.remove("LookAt")
    return all_action


def get_agent_node(nodes):
    agent_node = None
    for node in nodes:

        if node['class_name'] == 'character':
            agent_node = node

    assert agent_node is not None, "get_agent_info -> not find agent node"

    return agent_node


def get_close_id_items_id(edges, node_id):
    ret = []
    for edge in edges:
        from_id = edge['from_id']
        to_id = edge['to_id']
        relation_type = edge['relation_type']
        if from_id == node_id and relation_type == "CLOSE":
            ret.append(to_id)
    return ret


def get_same_state_class_nodes(nodes, states, class_name):
    ret = []
    for node in nodes:
        if class_name == node['class_name'] and states == node['states']:
            ret.append(node)

    return ret


def get_node_from_id(nodes, id):
    for node in nodes:
        if node['id'] == id:
            return node


def get_node_from_command(nodes, command):
    pattern = r'\((\d+)\)'
    matches = re.findall(pattern, command)
    ids = [int(match) for match in matches]
    return [get_node_from_id(nodes, id) for id in ids]
