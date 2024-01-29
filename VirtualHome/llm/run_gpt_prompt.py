import json
from .metrics import metrics
from .gpt_request import ChatGPT_safe_generate_response
from actions import actions_description, actions_instruction, instruction_to_action_name


def generate_prompt(curr_input, prompt_lib_file):
    """
  Takes in the current input (e.g. comment that you want to classifiy) and
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this
  function replaces this substr with the actual curr_input to produce the
  final promopt that will be sent to the GPT3 server.
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file.
  RETURNS:
    a str prompt that will be sent to OpenAI's GPT server.
  """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def run_gpt_daily_plan(room_interactive_item_str, test_input=None, verbose=False):
    def create_prompt_input(room_interactive_item_str):
        return [room_interactive_item_str]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data, list), "run_gpt_daily_plan -> json_data should be a list"
        assert 'activity' in json_data[0].keys(), "run_gpt_daily_plan -> json_data should has key: activity"
        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/daily_plan.txt"
    prompt_input = create_prompt_input(room_interactive_item_str)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_get_relative_items(task, relative_items, test_input=None, verbose=False):
    def create_prompt_input(task, relative_items):
        return [task, relative_items]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data['items'], list), "run_gpt_get_relative_items -> json_data['items'] should be a list"
        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/get_relative_items.txt"
    prompt_input = create_prompt_input(task, relative_items)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_get_relative_actions(task, room, agent_info, relative_items, test_input=None, verbose=False):
    def create_prompt_input(task, room, agent_info, relative_items):
        return [task, room, agent_info, relative_items]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data['actions'], list), "run_gpt_get_relative_actions -> json_data['actions'] should " \
                                                       "be a list."
        for action in json_data['actions']:
            if action not in actions_description.keys():
                assert False, f"run_gpt_get_relative_actions -> {action} not found"

        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/get_relative_action.txt"
    prompt_input = create_prompt_input(task, room, agent_info, relative_items)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_get_rough_actions(task, room, agent_info, relative_items, candidate_actions, test_input=None,
                              verbose=False):
    def create_prompt_input(task, room, agent_info, relative_items, candidate_actions):
        actions_description_prompt = ""
        actions_instruction_prompt = ""
        for action_name in candidate_actions:
            actions_description_prompt += f"{action_name}: {actions_description[action_name]}\n"
            actions_instruction_prompt += f"{action_name}: {actions_instruction[action_name]}\n"

        return [task, room, agent_info, relative_items, actions_description_prompt, actions_instruction_prompt]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data['commands'], list), "run_gpt_get_rough_actions -> json_data['commands'] should " \
                                                        "be a list."
        # for action in json_data['actions']:
        #     if action not in actions_description.keys():
        #         assert False, f"run_gpt_get_relative_actions -> {action} not found"

        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/get_rough_action.txt"
    prompt_input = create_prompt_input(task, room, agent_info, relative_items, candidate_actions)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_get_next_rough_actions(task, room, agent_info, relative_items, candidate_actions, previous_action,
                                   test_input=None,
                                   verbose=False):
    def create_prompt_input(task, room, agent_info, relative_items, candidate_actions, previous_action):
        actions_description_prompt = ""
        # actions_instruction_prompt = ""
        for action_name in candidate_actions:
            actions_description_prompt += f"{actions_instruction[action_name]}: {actions_description[action_name]}\n"
            # actions_instruction_prompt += f"{action_name}: {actions_instruction[action_name]}\n"

        previous_action_prompt = ""
        if len(previous_action) == 0:
            previous_action_prompt = "There are no previously executed actions; the current action being generated is " \
                                     "the first action to be executed. "
        else:
            previous_action_prompt = '\n'.join(previous_action)

        relative_items_properties_prompt = ""
        for k, v in relative_items.items():
            relative_items_properties_prompt += f"{k}: {','.join(v)}\n"

        return [task, room, agent_info, relative_items_properties_prompt, actions_description_prompt,
                previous_action_prompt]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data['command'], str), "run_gpt_get_next_rough_actions -> json_data['command'] should " \
                                                      "be a str."

        assert json_data['command'].split(' ')[
                   0] in instruction_to_action_name.keys(), "run_gpt_get_next_rough_actions " \
                                                            "-> illegal action"
        # if len(json_data['command'].split(' ')) > 0:
        #     for item in json_data['command'].split(' ')[1:]:
        #         assert item in relative_items.keys(), f"run_gpt_get_next_rough_actions " \
        #                                               f"-> illegal item{item}"

        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/get_next_rough_action.txt"
    prompt_input = create_prompt_input(task, room, agent_info, relative_items, candidate_actions, previous_action)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_get_next_rough_actions_with_forbidden_action(task, room, agent_info, relative_items, candidate_actions,
                                                         previous_action, forbidden_action,
                                                         test_input=None,
                                                         verbose=False):
    def create_prompt_input(task, room, agent_info, relative_items, candidate_actions, previous_action,
                            forbidden_action):
        actions_description_prompt = ""
        # actions_instruction_prompt = ""
        for action_name in candidate_actions:
            actions_description_prompt += f"{actions_instruction[action_name]}: {actions_description[action_name]}\n"
            # actions_instruction_prompt += f"{action_name}: {actions_instruction[action_name]}\n"

        previous_action_prompt = ""
        if len(previous_action) == 0:
            previous_action_prompt = "There are no previously executed actions; the current action being generated is " \
                                     "the first action to be executed. "
        else:
            previous_action_prompt = '\n'.join(previous_action)

        relative_items_properties_prompt = ""
        for k, v in relative_items.items():
            relative_items_properties_prompt += f"{k}: {','.join(v)}\n"

        forbidden_action_prompt = "\n".join(forbidden_action)

        return [task, room, agent_info, relative_items_properties_prompt, actions_description_prompt,
                previous_action_prompt, forbidden_action_prompt]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data['command'], str), "run_gpt_get_next_rough_actions -> json_data['command'] should " \
                                                      "be a str."

        assert json_data['command'].split(' ')[
                   0] in instruction_to_action_name.keys(), "run_gpt_get_next_rough_actions " \
                                                            "-> illegal action"
        # if len(json_data['command'].split(' ')) > 0:
        #     for item in json_data['command'].split(' ')[1:]:
        #         assert item in relative_items.keys() or item in , f"run_gpt_get_next_rough_actions " \
        #                                               f"-> illegal item {item}"

        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return {
            "command": "walk home"
        }

    prompt_template = "llm/prompt_template/get_next_rough_action_with_error.txt"
    prompt_input = create_prompt_input(task, room, agent_info, relative_items, candidate_actions, previous_action,
                                       forbidden_action)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_get_item_id(task, room, agent_info, previous_action, rough_command, item_name, candidate_items,
                        test_input=None,
                        verbose=False):
    def create_prompt_input(task, room, agent_info, previous_action, rough_command, item_name,
                            candidate_items):
        if len(previous_action) == 0:
            previous_action_prompt = "There are no previously executed actions; the current action being generated is " \
                                     "the first action to be executed. "
        else:
            previous_action_prompt = '\n'.join(previous_action)

        if " " not in rough_command:
            action_name = instruction_to_action_name[rough_command]
        else:
            action_name = instruction_to_action_name[rough_command.split(" ")[0]]

        current_action_description = f"{action_name}: {actions_description[action_name]}"

        return [task, room, agent_info, previous_action_prompt, current_action_description, rough_command, item_name,
                candidate_items]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data['id'], int), "run_gpt_get_item_id -> json_data['id'] should " \
                                                 "be a id<int>."
        # for action in json_data['actions']:
        #     if action not in actions_description.keys():
        #         assert False, f"run_gpt_get_relative_actions -> {action} not found"

        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/get_item_id.txt"
    prompt_input = create_prompt_input(task, room, agent_info, previous_action, rough_command, item_name,
                                       candidate_items)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output


def run_gpt_check_task_state(task, room, agent_info, previous_action,
                             test_input=None,
                             verbose=False):
    def create_prompt_input(task, room, agent_info, previous_action):
        previous_action_prompt = '\n'.join(previous_action)

        return [task, room, agent_info, previous_action_prompt]

    def __chat_func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert 'type' in json_data.keys(), "run_gpt_get_item_id -> json_data should have key {type}"
        assert 'reason' in json_data.keys(), "run_gpt_get_item_id -> json_data should have key {reason}"

        return json_data

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            __chat_func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = []
        return cleaned_dict

    prompt_template = "llm/prompt_template/check_task_state.txt"
    prompt_input = create_prompt_input(task, room, agent_info, previous_action)
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, verbose)
    return output
