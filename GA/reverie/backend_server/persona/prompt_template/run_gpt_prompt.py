"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: run_gpt_prompt.py
Description: Defines all run gpt prompt functions. These functions directly
interface with the safe_generate_response function.
"""
import json
import random
import re
import datetime
import sys
import ast

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *


def get_random_alphanumeric(i=6, j=6):
    """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
    k = random.randint(i, j)
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False):
    """
  Given the persona, returns an integer that indicates the hour when the 
  persona wakes up.  

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    integer for the wake up hour.
  """

    def create_prompt_input(persona, test_input=None):
        if test_input: return test_input
        prompt_input = [persona.scratch.get_str_iss(),
                        persona.scratch.get_str_lifestyle(),
                        persona.scratch.get_str_firstname()]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = int(gpt_response.strip().lower().split("am")[0])
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe():
        fs = 8
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 5,
                 "temperature": 0.8, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
    prompt_input = create_prompt_input(persona, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_daily_plan(persona,
                              wake_up_hour,
                              test_input=None,
                              verbose=False):
    """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form: 
  'wake up and complete the morning routine at 6:00 am', 
  'eat breakfast at 7:00 am',.. 
  Note that the actions come without a period. 

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    a list of daily actions in broad strokes.
  """

    def create_prompt_input(persona, wake_up_hour, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.name]
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 AM"]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data, list), "run_gpt_prompt_daily_plan -> gpt_response should be a list"
        daily_req = []
        n_m1_hourly_compressed = []

        curr_min = 0

        rest_time = 24 * 60
        n_m1_hourly_compressed += [["sleeping", int(wake_up_hour * 60) - curr_min]]
        rest_time -= int(wake_up_hour * 60) - curr_min

        for activity_json in json_data:
            activity = activity_json["activity"]
            start_time = datetime.datetime.strptime(activity_json["start"], '%I:%M %p')
            end_time = datetime.datetime.strptime(activity_json["end"], '%I:%M %p')
            daily_req.append(f"{activity} from {activity_json['start']} to {activity_json['end']}")
            min_diff_time = (end_time - start_time).total_seconds() / 60
            if min_diff_time < 0:
                min_diff_time = 1440 + min_diff_time

            n_m1_hourly_compressed += [[activity, int(min_diff_time)]]
            rest_time -= min_diff_time
            print(f"activity_json -> s:{start_time} e:{end_time} m:{min_diff_time} r:{rest_time} t:{activity}")

            # assert rest_time > 0, "generate_first_daily_plan -> rest time should larger than 0"
            assert min_diff_time > 0, "generate_first_daily_plan -> min_diff_time time should larger than 0"

        return json_data

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe():
        fs = []
        fs.append({"activity": "sleep", "hour": "06:00 AM"})
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 500,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                 __func_validate, __func_clean_up)
    # output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
    #           + output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_hourly_schedule(persona,
                                            start_hour,
                                            end_hour,
                                            test_input=None,
                                            verbose=False):
    def create_prompt_input(persona, start_hour, end_hour):
        daily_plan = ""
        for count, i in enumerate(persona.scratch.daily_req):
            daily_plan += f"{str(count + 1)}) {i}.\n"
        return [persona.scratch.get_str_firstname(), start_hour, end_hour, persona.scratch.get_str_iss(), daily_plan]

    def __func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data, list), "run_gpt_prompt_generate_hourly_schedule -> gpt_response should be a list"

        return json_data

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe():
        fs = []
        fs.append({"activity": "sleep", "hour": "06:00 AM"})
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.txt"
    prompt_input = create_prompt_input(persona, start_hour, end_hour)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                 __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_next_day_remember(persona,
                                     statement,
                                     test_input=None,
                                     verbose=False):
    def create_prompt_input(persona, statement):
        return [persona.scratch.get_str_firstname(), statement, persona.scratch.curr_time.strftime('%a %b %d')]

    def __func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data, list), "run_gpt_prompt_next_day_remember -> gpt_response should be a list"
        return json_data

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe():
        fs = []
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/lifestyle/next_day_remember.txt"
    prompt_input = create_prompt_input(persona, statement)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                 __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_next_day_plan(persona,
                                 wake_up_hour,
                                 plan_note,
                                 summary_thoughts,
                                 test_input=None,
                                 verbose=False):
    """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form:
  'wake up and complete the morning routine at 6:00 am',
  'eat breakfast at 7:00 am',..
  Note that the actions come without a period.

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    a list of daily actions in broad strokes.
  """

    def create_prompt_input(persona, wake_up_hour, plan_note, summary_thoughts, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.name]
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 AM"]
        prompt_input += [plan_note]
        prompt_input += [summary_thoughts]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data, list), "run_gpt_prompt_next_day_plan -> gpt_response should be a list"
        daily_req = []
        n_m1_hourly_compressed = []

        curr_min = 0

        rest_time = 24 * 60
        n_m1_hourly_compressed += [["sleeping", int(wake_up_hour * 60) - curr_min]]
        rest_time -= int(wake_up_hour * 60) - curr_min

        for activity_json in json_data:
            activity = activity_json["activity"]
            start_time = datetime.datetime.strptime(activity_json["start"], '%I:%M %p')
            end_time = datetime.datetime.strptime(activity_json["end"], '%I:%M %p')
            daily_req.append(f"{activity} from {activity_json['start']} to {activity_json['end']}")
            min_diff_time = (end_time - start_time).total_seconds() / 60
            if min_diff_time < 0:
                min_diff_time = 1440 + min_diff_time

            n_m1_hourly_compressed += [[activity, int(min_diff_time)]]
            rest_time -= min_diff_time
            print(f"activity_json -> s:{start_time} e:{end_time} m:{min_diff_time} r:{rest_time} t:{activity}")

            # assert rest_time > 0, "generate_first_daily_plan -> rest time should larger than 0"
            assert min_diff_time > 0, "generate_first_daily_plan -> min_diff_time time should larger than 0"

        return json_data

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe():
        fs = []
        fs.append({"activity": "sleep", "hour": "06:00 AM"})
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 500,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/lifestyle/next_day.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, plan_note, summary_thoughts, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                 __func_validate, __func_clean_up)
    # output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
    #           + output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_task_decomp(persona,
                               task,
                               duration,
                               test_input=None,
                               verbose=False):
    def create_prompt_input(persona, task, duration, test_input=None):

        """
    Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is 
    planning on sleeping, 06:00 ~ 07:00am, Maeve is 
    planning on waking up and doing her morning routine, 
    and from 07:00am ~08:00am, Maeve is planning on having breakfast.  
    """

        curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
        all_indices = []
        # if curr_f_org_index > 0:
        #   all_indices += [curr_f_org_index-1]
        all_indices += [curr_f_org_index]
        if curr_f_org_index + 1 <= len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 1]
        if curr_f_org_index + 2 <= len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 2]

        curr_time_range = ""

        print("DEBUG")
        print(persona.scratch.f_daily_schedule_hourly_org)
        print(all_indices)

        summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
        summ_str += f'From '
        for index in all_indices:
            print("index", index)
            if index < len(persona.scratch.f_daily_schedule_hourly_org):
                start_min = 0
                for i in range(index):
                    start_min += persona.scratch.f_daily_schedule_hourly_org[i][1]
                end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
                start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                              + datetime.timedelta(minutes=start_min))
                end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S")
                            + datetime.timedelta(minutes=end_min))
                start_time_str = start_time.strftime("%H:%M%p")
                end_time_str = end_time.strftime("%H:%M%p")
                summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
                if curr_f_org_index + 1 == index:
                    curr_time_range = f'{start_time_str} ~ {end_time_str}'
        summ_str = summ_str[:-2] + "."

        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [summ_str]
        # prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [task]
        prompt_input += [curr_time_range]
        prompt_input += [duration]
        prompt_input += [persona.scratch.get_str_firstname()]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print("TOODOOOOOO")
        print(gpt_response)
        print("-==- -==- -==- ")

        # TODO SOMETHING HERE sometimes fails... See screenshot
        temp = [i.strip() for i in gpt_response.split("\n")]
        _cr = []
        cr = []
        for count, i in enumerate(temp):
            if count != 0:
                _cr += [" ".join([j.strip() for j in i.split(" ")][3:])]
            else:
                _cr += [i]
        for count, i in enumerate(_cr):
            k = [j.strip() for j in i.split("(duration in minutes:")]
            task = k[0]
            if task[-1] == ".":
                task = task[:-1]
            duration = int(k[1].split(",")[0].strip())
            cr += [[task, duration]]

        total_expected_min = int(prompt.split("(total duration in minutes")[-1]
                                 .split("):")[0].strip())

        # TODO -- now, you need to make sure that this is the same as the sum of
        #         the current action sequence.
        curr_min_slot = [["dummy", -1], ]  # (task_name, task_index)
        for count, i in enumerate(cr):
            i_task = i[0]
            i_duration = i[1]

            i_duration -= (i_duration % 5)
            if i_duration > 0:
                for j in range(i_duration):
                    curr_min_slot += [(i_task, count)]
        curr_min_slot = curr_min_slot[1:]

        if len(curr_min_slot) > total_expected_min:
            last_task = curr_min_slot[60]
            for i in range(1, 6):
                curr_min_slot[-1 * i] = last_task
        elif len(curr_min_slot) < total_expected_min:
            last_task = curr_min_slot[-1]
            for i in range(total_expected_min - len(curr_min_slot)):
                curr_min_slot += [last_task]

        cr_ret = [["dummy", -1], ]
        for task, task_index in curr_min_slot:
            if task != cr_ret[-1][0]:
                cr_ret += [[task, 1]]
            else:
                cr_ret[-1][1] += 1
        cr = cr_ret[1:]

        return cr

    def __func_validate(gpt_response, prompt=""):
        # TODO -- this sometimes generates error
        try:
            __func_clean_up(gpt_response, prompt)
        except Exception as e:
            metrics.fail_record(e)
            return False
        return gpt_response

    def get_fail_safe():
        fs = ["asleep"]
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
    prompt_input = create_prompt_input(persona, task, duration)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    print("?????")
    print(prompt)
    output = safe_generate_response(prompt, gpt_param, 5, get_fail_safe(),
                                    __func_validate, __func_clean_up)

    # TODO THERE WAS A BUG HERE...
    # This is for preventing overflows...
    """
  File "/Users/joonsungpark/Desktop/Stanford/Projects/
  generative-personas/src_exploration/reverie_simulation/
  brain/get_next_action_v3.py", line 364, in run_gpt_prompt_task_decomp
  fin_output[-1][1] += (duration - ftime_sum)
  IndexError: list index out of range
  """

    print("IMPORTANT VVV DEBUG")

    # print (prompt_input)
    # print (prompt)
    print(output)

    fin_output = []
    time_sum = 0
    for i_task, i_duration in output:
        time_sum += i_duration
        # HM?????????
        # if time_sum < duration:
        if time_sum <= duration:
            fin_output += [[i_task, i_duration]]
        else:
            break
    ftime_sum = 0
    for fi_task, fi_duration in fin_output:
        ftime_sum += fi_duration

    # print ("for debugging... line 365", fin_output)
    fin_output[-1][1] += (duration - ftime_sum)
    output = fin_output

    task_decomp = output
    ret = []
    for decomp_task, duration in task_decomp:
        ret += [[f"{task} ({decomp_task})", duration]]
    output = ret

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_action_sector(action_description,
                                 persona,
                                 maze,
                                 test_input=None,
                                 verbose=False):
    def create_prompt_input(action_description, persona, maze, test_input=None):
        prompt_input = []

        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [persona.scratch.living_area.split(":")[1]]
        x = f"{act_world}:{persona.scratch.living_area.split(':')[1]}"
        prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

        prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
        x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)]

        if persona.scratch.get_str_daily_plan_req() != "":
            prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}"]
        else:
            prompt_input += [""]

        # MAR 11 TEMP

        # END MAR 11 TEMP

        action_description_1 = action_description
        action_description_2 = action_description
        if "(" in action_description:
            action_description_1 = action_description.split("(")[0].strip()
            action_description_2 = action_description.split("(")[-1][:-1]
        prompt_input += [action_description_1]
        prompt_input += [action_description_2]

        accessible_sector_str = ""
        for i, sector in enumerate(fin_accessible_sectors):
            accessible_sector_str += f"{i + 1}. {sector}\n"
        prompt_input += [accessible_sector_str]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cleaned_response = json.loads(gpt_response)["output"]

        assert cleaned_response in accessible_sector_str, f"{cleaned_response} should be in {accessible_sector_str}"
        return cleaned_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)
        return False

    def get_fail_safe():
        fs = ("kitchen")
        return fs

    # # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   cr = gpt_response.strip()
    #   return cr

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try:
    #     gpt_response = __func_clean_up(gpt_response, prompt="")
    #   except Exception as e:
    #     metrics.fail_record(e)
    #     return False
    #   return True

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 20") ########
    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/action_location_sector_v2.txt" ########
    # prompt_input = create_prompt_input(action_description, persona, maze)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = "Johnson Park" ########
    # special_instruction = "The value for the output must contain one of the area options above verbatim (including lower/upper case)." ########
    # fail_safe = get_fail_safe() ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/lifestyle/action_location_sector.txt"

    act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
    curr = accessible_sector_str.split(", ")
    fin_accessible_sectors = []
    for i in curr:
        if "'s house" in i:
            if persona.scratch.last_name in i:
                fin_accessible_sectors += [i]
        else:
            fin_accessible_sectors += [i]

    prompt_input = create_prompt_input(action_description, persona, maze)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
    if output not in x:
        # output = random.choice(x)
        output = persona.scratch.living_area.split(":")[1]

    print("DEBUG", random.choice(x), "------", output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_action_arena(action_description,
                                persona,
                                maze, act_world, act_sector,
                                test_input=None,
                                verbose=False):
    def create_prompt_input(action_description, persona, maze, act_world, act_sector, test_input=None):
        prompt_input = []
        # prompt_input += [persona.scratch.get_str_name()]
        # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
        # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["sector"]]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [act_sector]

        # MAR 11 TEMP
        # END MAR 11 TEMP
        if len(fin_accessible_arenas) == 0:
            print(f"{persona.name} tree:{persona.s_mem.print_tree()}")
            assert False, "run_gpt_prompt_action_arena -> wrong accessible_arena_str"
        accessible_arena_str = ''
        for i, arena in enumerate(fin_accessible_arenas):
            accessible_arena_str += f"{i + 1}. {arena}\n"

        prompt_input += [accessible_arena_str]

        action_description_1 = action_description
        action_description_2 = action_description
        if "(" in action_description:
            action_description_1 = action_description.split("(")[0].strip()
            action_description_2 = action_description.split("(")[-1][:-1]
        prompt_input += [action_description_1]
        prompt_input += [action_description_2]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cleaned_response = json.loads(gpt_response)['output']

        assert cleaned_response in fin_accessible_arenas, f"{cleaned_response} should be in {fin_accessible_arenas}"

        return cleaned_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(e)

        return False

    def get_fail_safe():
        return random.choice(fin_accessible_arenas)

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/lifestyle/action_location_arena.txt"

    x = f"{act_world}:{act_sector}"
    accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
    curr = accessible_arena_str.split(", ")
    fin_accessible_arenas = []
    for i in curr:
        if "'s room" in i:
            if persona.scratch.last_name in i:
                fin_accessible_arenas += [i]
        else:
            fin_accessible_arenas += [i]

    prompt_input = create_prompt_input(action_description, persona, maze, act_world, act_sector)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    print(output)
    # y = f"{act_world}:{act_sector}"
    # x = [i.strip() for i in persona.s_mem.get_str_accessible_sector_arenas(y).split(",")]
    # if output not in x:
    #   output = random.choice(x)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_action_game_object(action_description,
                                      persona,
                                      maze,
                                      temp_address,
                                      test_input=None,
                                      verbose=False):
    def create_prompt_input(action_description,
                            persona,
                            temp_address,
                            test_input=None):
        prompt_input = []
        if "(" in action_description:
            action_description = action_description.split("(")[-1][:-1]

        prompt_input += [action_description]
        prompt_input += [persona
                             .s_mem.get_str_accessible_arena_game_objects(temp_address)]
        return prompt_input

    def __func_validate(gpt_response, prompt=""):
        if len(gpt_response.strip()) < 1:
            return False
        return True

    def __func_clean_up(gpt_response, prompt=""):
        cleaned_response = gpt_response.strip()
        return cleaned_response

    def get_fail_safe():
        fs = ("bed")
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v1/action_object_v2.txt"
    prompt_input = create_prompt_input(action_description,
                                       persona,
                                       temp_address,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
    if output not in x:
        output = random.choice(x)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False):
    def create_prompt_input(action_description):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [action_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip()
        if len(cr) > 3:
            cr = cr[:3]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) == 0:
                return False
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe():
        fs = "😋"
        return fs

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        cr = gpt_response.strip()
        if len(cr) > 3:
            cr = cr[:3]
        return cr

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) == 0:
                return False
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 4")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt"  ########
    prompt_input = create_prompt_input(action_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "🛁🧖‍♀️"  ########
    special_instruction = "The value for the output must ONLY contain the emojis."  ########
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    # prompt_template = "persona/prompt_template/v2/generate_pronunciatio_v1.txt"
    # prompt_input = create_prompt_input(action_description)

    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_event_triple(action_description, persona, verbose=False):
    def create_prompt_input(action_description, persona):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [persona.name,
                        action_description,
                        persona.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip()
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) != 2:
                return False
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe(persona):
        fs = (persona.name, "is", "idle")
        return fs

    # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   cr = gpt_response.strip()
    #   cr = [i.strip() for i in cr.split(")")[0].split(",")]
    #   return cr

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try:
    #     gpt_response = __func_clean_up(gpt_response, prompt="")
    #     if len(gpt_response) != 2:
    #       return False
    #   except Exception as e:
    #     metrics.fail_record(e)return False
    #   return True

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 5") ########
    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_event_triple_v1.txt" ########
    # prompt_input = create_prompt_input(action_description, persona)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = "(Jane Doe, cooking, breakfast)" ########
    # special_instruction = "The value for the output must ONLY contain the triple. If there is an incomplete element, just say 'None' but there needs to be three elements no matter what." ########
    # fail_safe = get_fail_safe(persona) ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 30,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
    prompt_input = create_prompt_input(action_description, persona)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(persona)  ########
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    output = (persona.name, output[0], output[1])

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False):
    def create_prompt_input(act_game_object, act_desp, persona):
        prompt_input = [act_game_object,
                        persona.name,
                        act_desp,
                        act_game_object,
                        act_game_object]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip()
        if cr[-1] == ".": cr = cr[:-1]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe(act_game_object):
        fs = f"{act_game_object} is idle"
        return fs

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        cr = gpt_response.strip()
        if cr[-1] == ".": cr = cr[:-1]
        return cr

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 6")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt"  ########
    prompt_input = create_prompt_input(act_game_object, act_desp, persona)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "being fixed"  ########
    special_instruction = "The output should ONLY contain the phrase that should go in <fill in>."  ########
    fail_safe = get_fail_safe(act_game_object)  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 30,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    # prompt_template = "persona/prompt_template/v2/generate_obj_event_v1.txt"
    # prompt_input = create_prompt_input(act_game_object, act_desp, persona)
    # prompt = generate_prompt(prompt_input, prompt_template)
    # fail_safe = get_fail_safe(act_game_object)
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False):
    def create_prompt_input(act_game_object, act_obj_desc):
        prompt_input = [act_game_object,
                        act_obj_desc,
                        act_game_object]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip()
        cr = [i.strip() for i in cr.split(")")[0].split(",")]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) != 2:
                return False
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe(act_game_object):
        fs = (act_game_object, "is", "idle")
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 30,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
    prompt_input = create_prompt_input(act_game_object, act_obj_desc)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(act_game_object)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    output = (act_game_object, output[0], output[1])

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_new_decomp_schedule(persona,
                                       hourly_schedule,
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None,
                                       verbose=False):
    def create_prompt_input(persona,
                            hourly_schedule,
                            main_act_dur,
                            truncated_act_dur,
                            start_time_hour,
                            end_time_hour,
                            inserted_act,
                            inserted_act_dur,
                            test_input=None):
        persona_name = persona.name
        start_hour_str = start_time_hour.strftime("%H:%M %p")
        end_hour_str = end_time_hour.strftime("%H:%M %p")

        original_plan = ""
        for_time = start_time_hour
        for i in main_act_dur:
            original_plan += f'{for_time.strftime("%H:%M %p")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M %p")} -- ' + \
                             i[0]
            original_plan += "\n"
            for_time += datetime.timedelta(minutes=int(i[1]))

        new_plan_init = ""
        for_time = start_time_hour
        for count, i in enumerate(truncated_act_dur):
            new_plan_init += f'{for_time.strftime("%H:%M %p")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M %p")} -- ' + \
                             i[0]
            new_plan_init += "\n"
            if count < len(truncated_act_dur) - 1:
                for_time += datetime.timedelta(minutes=int(i[1]))

        new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M %p") + " ~"

        truncated_act_data = truncated_act_dur[-1][0]
        if len(truncated_act_dur) >= 2:
            truncated_act_data = truncated_act_dur[-2][0]

        prompt_input = [persona_name,
                        start_hour_str,
                        end_hour_str,
                        original_plan,
                        truncated_act_data,
                        inserted_act,
                        inserted_act_dur,
                        (for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%H:%M %p"),
                        hourly_schedule,
                        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        json_data = json.loads(gpt_response)
        assert isinstance(json_data, list), "run_gpt_prompt_new_decomp_schedule -> gpt_response should be a list"
        if len(json_data) > 0:
            assert "activity" in json_data[0].keys(), "run_gpt_prompt_new_decomp_schedule -> no key - activity"
            assert "start" in json_data[0].keys(), "run_gpt_prompt_new_decomp_schedule -> no key - start"
            assert "end" in json_data[0].keys(), "run_gpt_prompt_new_decomp_schedule -> no key - end"
        return json_data

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    def get_fail_safe(main_act_dur, truncated_act_dur):
        dur_sum = 0
        for act, dur in main_act_dur: dur_sum += dur

        ret = truncated_act_dur[:]
        ret += main_act_dur[len(ret) - 1:]

        # If there are access, we need to trim...
        ret_dur_sum = 0
        count = 0
        over = None
        for act, dur in ret:
            ret_dur_sum += dur
            if ret_dur_sum == dur_sum:
                break
            if ret_dur_sum > dur_sum:
                over = ret_dur_sum - dur_sum
                break
            count += 1

        if over:
            ret = ret[:count + 1]
            ret[-1][1] -= over

        _ret = []
        for single_time in ret:
            _ret.append(
                {
                    "activity": single_time[0],
                    "start": "05:00 AM",
                    "end": (datetime.datetime.strptime("05:00 AM", '%H:%M %p') +
                            datetime.timedelta(minutes=single_time[1])).strftime('%H:%M %p')
                }
            )

        return _ret

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/lifestyle/new_decomp_schedule.txt"
    prompt_input = create_prompt_input(persona,
                                       hourly_schedule,
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    # print ("* * * * output")
    # print (output)
    # print ('* * * * fail_safe')
    # print (fail_safe)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved, test_input=None,
                                  verbose=False):
    def create_prompt_input(init_persona, target_persona, retrieved,
                            test_input=None):
        last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
        last_chatted_time = ""
        last_chat_about = ""
        if last_chat:
            last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
            last_chat_about = last_chat.description

        context = ""
        for c_node in retrieved["events"]:
            curr_desc = c_node.description.split(" ")
            curr_desc[2:3] = ["was"]
            curr_desc = " ".join(curr_desc)
            context += f"{curr_desc}. "
        context += "\n"
        for c_node in retrieved["thoughts"]:
            context += f"{c_node.description}. "

        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
        init_act_desc = init_persona.scratch.act_description
        if "(" in init_act_desc:
            init_act_desc = init_act_desc.split("(")[-1][:-1]

        if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
            init_p_desc = f"{init_persona.name} is already {init_act_desc}"
        elif "waiting" in init_act_desc:
            init_p_desc = f"{init_persona.name} is {init_act_desc}"
        else:
            init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

        target_act_desc = target_persona.scratch.act_description
        if "(" in target_act_desc:
            target_act_desc = target_act_desc.split("(")[-1][:-1]

        if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
            target_p_desc = f"{target_persona.name} is already {target_act_desc}"
        elif "waiting" in init_act_desc:
            target_p_desc = f"{init_persona.name} is {init_act_desc}"
        else:
            target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"

        prompt_input = []
        prompt_input += [context]

        prompt_input += [curr_time]

        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]
        prompt_input += [last_chatted_time]
        prompt_input += [last_chat_about]

        prompt_input += [init_p_desc]
        prompt_input += [target_p_desc]
        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]
        return prompt_input

    def __func_validate(gpt_response, prompt=""):
        try:
            if gpt_response.split("Answer in yes or no:")[-1].strip().lower() in ["yes", "no"]:
                return True
            return False
        except Exception as e:
            metrics.fail_record(e)
            return False

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split("Answer in yes or no:")[-1].strip().lower()

    def get_fail_safe():
        fs = "yes"
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.txt"
    prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved, test_input=None,
                                   verbose=False):
    def create_prompt_input(init_persona, target_persona, retrieved,
                            test_input=None):

        context = ""
        for c_node in retrieved["events"]:
            curr_desc = c_node.description.split(" ")
            curr_desc[2:3] = ["was"]
            curr_desc = " ".join(curr_desc)
            context += f"{curr_desc}. "
        context += "\n"
        for c_node in retrieved["thoughts"]:
            context += f"{c_node.description}. "

        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
        init_act_desc = init_persona.scratch.act_description
        if "(" in init_act_desc:
            init_act_desc = init_act_desc.split("(")[-1][:-1]
        if len(init_persona.scratch.planned_path) == 0:
            loc = ""
            if ":" in init_persona.scratch.act_address:
                loc = init_persona.scratch.act_address.split(":")[-1] + " in " + \
                      init_persona.scratch.act_address.split(":")[-2]
            init_p_desc = f"{init_persona.name} is already {init_act_desc} at {loc}"
        else:
            loc = ""
            if ":" in init_persona.scratch.act_address:
                loc = init_persona.scratch.act_address.split(":")[-1] + " in " + \
                      init_persona.scratch.act_address.split(":")[-2]
            init_p_desc = f"{init_persona.name} is on the way to {init_act_desc} at {loc}"

        target_act_desc = target_persona.scratch.act_description
        if "(" in target_act_desc:
            target_act_desc = target_act_desc.split("(")[-1][:-1]
        if len(target_persona.scratch.planned_path) == 0:
            loc = ""
            if ":" in target_persona.scratch.act_address:
                loc = target_persona.scratch.act_address.split(":")[-1] + " in " + \
                      target_persona.scratch.act_address.split(":")[-2]
            target_p_desc = f"{target_persona.name} is already {target_act_desc} at {loc}"
        else:
            loc = ""
            if ":" in target_persona.scratch.act_address:
                loc = target_persona.scratch.act_address.split(":")[-1] + " in " + \
                      target_persona.scratch.act_address.split(":")[-2]
            target_p_desc = f"{target_persona.name} is on the way to {target_act_desc} at {loc}"

        prompt_input = []
        prompt_input += [context]
        prompt_input += [curr_time]
        prompt_input += [init_p_desc]
        prompt_input += [target_p_desc]

        prompt_input += [init_persona.name]
        prompt_input += [init_act_desc]
        prompt_input += [target_persona.name]
        prompt_input += [target_act_desc]

        prompt_input += [init_act_desc]
        return prompt_input

    def __func_validate(gpt_response, prompt=""):
        try:
            if gpt_response.split("Answer: Option")[-1].strip().lower() in ["3", "2", "1"]:
                return True
            return False
        except Exception as e:
            metrics.fail_record(e)
            return False

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split("Answer: Option")[-1].strip().lower()

    def get_fail_safe():
        fs = "3"
        return fs

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
    prompt_input = create_prompt_input(persona, target_persona, retrieved,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False):
    def create_prompt_input(init_persona, target_persona, curr_loc,
                            test_input=None):

        prev_convo_insert = "\n"
        if init_persona.a_mem.seq_chat:
            for i in init_persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((init_persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation.\n'
                    for row in i.filling:
                        prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if init_persona.a_mem.seq_chat:
            if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[
                -1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""

        init_persona_thought_nodes = init_persona.a_mem.retrieve_relevant_thoughts(target_persona.scratch.act_event[0],
                                                                                   target_persona.scratch.act_event[1],
                                                                                   target_persona.scratch.act_event[2])
        init_persona_thought = ""
        for i in init_persona_thought_nodes:
            init_persona_thought += f"-- {i.description}\n"

        target_persona_thought_nodes = target_persona.a_mem.retrieve_relevant_thoughts(
            init_persona.scratch.act_event[0],
            init_persona.scratch.act_event[1],
            init_persona.scratch.act_event[2])
        target_persona_thought = ""
        for i in target_persona_thought_nodes:
            target_persona_thought += f"-- {i.description}\n"

        init_persona_curr_desc = ""
        if init_persona.scratch.planned_path:
            init_persona_curr_desc = f"{init_persona.name} is on the way to {init_persona.scratch.act_description}"
        else:
            init_persona_curr_desc = f"{init_persona.name} is {init_persona.scratch.act_description}"

        target_persona_curr_desc = ""
        if target_persona.scratch.planned_path:
            target_persona_curr_desc = f"{target_persona.name} is on the way to {target_persona.scratch.act_description}"
        else:
            target_persona_curr_desc = f"{target_persona.name} is {target_persona.scratch.act_description}"

        curr_loc = curr_loc["arena"]

        prompt_input = []
        prompt_input += [init_persona.scratch.get_str_iss()]
        prompt_input += [target_persona.scratch.get_str_iss()]

        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]
        prompt_input += [init_persona_thought]

        prompt_input += [target_persona.name]
        prompt_input += [init_persona.name]
        prompt_input += [target_persona_thought]

        prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S")]

        prompt_input += [init_persona_curr_desc]
        prompt_input += [target_persona_curr_desc]

        prompt_input += [prev_convo_insert]

        prompt_input += [init_persona.name]
        prompt_input += [target_persona.name]

        prompt_input += [curr_loc]
        prompt_input += [init_persona.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # print ("???")
        # print (gpt_response)

        gpt_response = (prompt + gpt_response).split("What would they talk about now?")[-1].strip()
        content = re.findall('"([^"]*)"', gpt_response)

        speaker_order = []
        for i in gpt_response.split("\n"):
            name = i.split(":")[0].strip()
            if name:
                speaker_order += [name]

        ret = []
        for count, speaker in enumerate(speaker_order):
            ret += [[speaker, content[count]]]

        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe(init_persona, target_persona):
        convo = [[init_persona.name, "Hi!"],
                 [target_persona.name, "Hi!"]]
        return convo

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 1000,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
    prompt_input = create_prompt_input(persona, target_persona, curr_loc,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(persona, target_persona)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False):
    def create_prompt_input(conversation, test_input=None):
        convo_str = ""
        for row in conversation:
            convo_str += f'{row[0]}: "{row[1]}"\n'

        prompt_input = [convo_str]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        ret = gpt_response.strip()
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "conversing with a housemate about morning greetings"

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        ret = gpt_response.strip()
        return ret

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 11")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt"  ########
    prompt_input = create_prompt_input(conversation, test_input)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "conversing about what to eat for lunch"  ########
    special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_conversation_v1.txt"
    # prompt_input = create_prompt_input(conversation, test_input)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False):
    def create_prompt_input(description, test_input=None):
        if "\n" in description:
            description = description.replace("\n", " <LINE_BREAK> ")
        prompt_input = [description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print("???")
        print(gpt_response)
        gpt_response = gpt_response.strip().split("Emotive keywords:")
        factual = [i.strip() for i in gpt_response[0].split(",")]
        emotive = [i.strip() for i in gpt_response[1].split(",")]
        all_keywords = factual + emotive
        ret = []
        for i in all_keywords:
            if i:
                i = i.lower()
                if i[-1] == ".":
                    i = i[:-1]
                ret += [i]
        print(ret)
        return set(ret)

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return []

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
    prompt_input = create_prompt_input(description, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False):
    def create_prompt_input(persona, keyword, concept_summary, test_input=None):
        prompt_input = [keyword, concept_summary, persona.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip()
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return ""

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 40,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
    prompt_input = create_prompt_input(persona, keyword, concept_summary)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_convo_to_thoughts(persona,
                                     init_persona_name,
                                     target_persona_name,
                                     convo_str,
                                     fin_target, test_input=None, verbose=False):
    def create_prompt_input(init_persona_name,
                            target_persona_name,
                            convo_str,
                            fin_target, test_input=None):
        prompt_input = [init_persona_name,
                        target_persona_name,
                        convo_str,
                        init_persona_name,
                        fin_target]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip()
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return ""

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 40,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
    prompt_input = create_prompt_input(init_persona_name,
                                       target_persona_name,
                                       convo_str,
                                       fin_target)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# TODO run_gpt_prompt_event_poignancy pool
def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False):
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = int(gpt_response.strip())
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return 4

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        gpt_response = int(gpt_response)
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 7")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt"  ########
    prompt_input = create_prompt_input(persona, event_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"  ########
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 3,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/poignancy_event_v1.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False):
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = int(gpt_response.strip())
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return 4

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        gpt_response = int(gpt_response)
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 8")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt"  ########
    prompt_input = create_prompt_input(persona, event_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"  ########
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 3,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/poignancy_thought_v1.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False):
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = int(gpt_response.strip())
        return gpt_response

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return 4

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        gpt_response = int(gpt_response)
        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 9")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt"  ########
    prompt_input = create_prompt_input(persona, event_description)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"  ########
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 3,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/poignancy_chat_v1.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False):
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = "1) " + gpt_response.strip()
        ret = []
        for i in gpt_response.split("\n"):
            ret += [i.split(") ")[-1]]
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe(n):
        return ["Who am I"] * n

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        if isinstance(gpt_response, list):
            return gpt_response
        ret = ast.literal_eval(gpt_response)
        return ret

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __chat_func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 12")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt"  ########
    prompt_input = create_prompt_input(persona, statements, n)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]'  ########
    special_instruction = "Output must be a list of str."  ########
    fail_safe = get_fail_safe(n)  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 150,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/generate_focal_pt_v1.txt"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(n)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False):
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        json_list = json.loads(gpt_response)
        ret = dict()
        for json_info in json_list:
            thought = json_info["insight"]
            related_index = json_info["related_index"]
            ret[thought] = related_index
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe(n):
        return ["I am hungry"] * n

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 500,
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(n)
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None,
                                              verbose=False):
    def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None):
        prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently,
                        statements, persona.scratch.name, target_persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.split('"')[0].strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 17")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt"  ########
    prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project'  ########
    special_instruction = 'The output should be a string that responds to the question.'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 150,
    #              "temperature": 0.5, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_chat_ideas_v1.txt"
    # prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# TODO run_gpt_prompt_agent_chat_summarize_relationship refine
def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None,
                                                     verbose=False):
    def create_prompt_input(persona, target_persona, statements, test_input=None):
        prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.split('"')[0].strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 18")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt"  ########
    prompt_input = create_prompt_input(persona, target_persona, statements)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project'  ########
    special_instruction = 'The output should be a string that responds to the question.'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 150,
    #              "temperature": 0.5, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_chat_relationship_v1.txt"
    # prompt_input = create_prompt_input(persona, target_persona, statements)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                              curr_context,
                              init_summ_idea,
                              target_summ_idea, test_input=None, verbose=False):
    def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None):
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        prompt_input = [persona.scratch.currently,
                        target_persona.scratch.currently,
                        prev_convo_insert,
                        curr_context,
                        curr_location,

                        persona.scratch.name,
                        init_summ_idea,
                        persona.scratch.name,
                        target_persona.scratch.name,

                        target_persona.scratch.name,
                        target_summ_idea,
                        target_persona.scratch.name,
                        persona.scratch.name,

                        persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        print(gpt_response)

        gpt_response = (prompt + gpt_response).split("Here is their conversation.")[-1].strip()
        content = re.findall('"([^"]*)"', gpt_response)

        speaker_order = []
        for i in gpt_response.split("\n"):
            name = i.split(":")[0].strip()
            if name:
                speaker_order += [name]

        ret = []
        for count, speaker in enumerate(speaker_order):
            ret += [[speaker, content[count]]]

        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        # ret = ast.literal_eval(gpt_response)

        print("a;dnfdap98fh4p9enf HEREE!!!")
        for row in gpt_response:
            print(row)

        return gpt_response

    def __chat_func_validate(gpt_response, prompt=""):  ############
        return True

    # print ("HERE JULY 23 -- ----- ") ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt"  ########
    prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea,
                                       target_summ_idea)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]'  ########
    special_instruction = 'The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"].'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    # print ("HERE END JULY 23 -- ----- ") ########
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 2000,
    #              "temperature": 0.7, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/agent_chat_v1.txt"
    # prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# =======================
# =======================
# =======================
# =======================


def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False):
    def create_prompt_input(persona, statements, question, test_input=None):
        prompt_input = [statements, persona.scratch.name, question]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.split('"')[0].strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 16")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt"  ########
    prompt_input = create_prompt_input(persona, statements, question)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project'  ########
    special_instruction = 'The output should be a string that responds to the question.'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 150,
    #              "temperature": 0.5, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v2/summarize_ideas_v1.txt"
    # prompt_input = create_prompt_input(persona, statements, question)
    # prompt = generate_prompt(prompt_input, prompt_template)

    # fail_safe = get_fail_safe()
    # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
    #                                  __func_validate, __func_clean_up)

    # if debug or verbose:
    #   print_run_prompts(prompt_template, persona, gpt_param,
    #                     prompt_input, prompt, output)

    # return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None,
                                            verbose=False):
    def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        interlocutor_desc,
                        prev_convo,
                        persona.scratch.name,
                        retrieved_summary,
                        persona.scratch.name, ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    # # ChatGPT Plugin ===========================================================
    # def __chat_func_clean_up(gpt_response, prompt=""): ############
    #   return gpt_response.split('"')[0].strip()

    # def __chat_func_validate(gpt_response, prompt=""): ############
    #   try:
    #     __func_clean_up(gpt_response, prompt)
    #     return True
    #   except Exception as e:
    #     return False

    # print ("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15") ########
    # gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
    #              "temperature": 0, "top_p": 1, "stream": False,
    #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # prompt_template = "persona/prompt_template/v3_ChatGPT/generate_next_convo_line_v1.txt" ########
    # prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)  ########
    # prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = 'Hello' ########
    # special_instruction = 'The output should be a string that responds to the question. Again, only use the context included in the "Note" to generate the response' ########
    # fail_safe = get_fail_safe() ########
    # output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
    #                                         __chat_func_validate, __chat_func_clean_up, True)
    # if output != False:
    #   return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 250,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
    prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False):
    def create_prompt_input(persona, whisper, test_input=None):
        prompt_input = [persona.scratch.name, whisper]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
    prompt_input = create_prompt_input(persona, whisper)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False):
    def create_prompt_input(persona, all_utt, test_input=None):
        prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
    prompt_input = create_prompt_input(persona, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False):
    def create_prompt_input(persona, all_utt, test_input=None):
        prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        return gpt_response.split('"')[0].strip()

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================
    def __chat_func_clean_up(gpt_response, prompt=""):  ############
        return gpt_response.strip()

    def __chat_func_validate(gpt_response, prompt=""):  ############
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 15")  ########
    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 15,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt"  ########
    prompt_input = create_prompt_input(persona, all_utt)  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe was interesting to talk to.'  ########
    special_instruction = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed'  ########
    fail_safe = get_fail_safe()  ########
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/prompt_template/v2/memo_on_convo_v1.txt"
    prompt_input = create_prompt_input(persona, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False):
    def create_prompt_input(comment, test_input=None):
        prompt_input = [comment]
        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = json.loads(gpt_response)
        return gpt_response["output"]

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            fields = ["output"]
            response = json.loads(gpt_response)
            for field in fields:
                if field not in response:
                    return False
            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        return None

    print("11")
    prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt"
    prompt_input = create_prompt_input(comment)
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def extract_first_json_dict(data_str):
    # Find the first occurrence of a JSON object within the string
    start_idx = data_str.find('{')
    end_idx = data_str.find('}', start_idx) + 1

    # Check if both start and end indices were found
    if start_idx == -1 or end_idx == 0:
        return None

    # Extract the first JSON dictionary
    json_str = data_str[start_idx:end_idx]

    try:
        # Attempt to parse the JSON data
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError as e:
        metrics.fail_record(e)
        # If parsing fails, return None
        return None


def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                        test_input=None, verbose=False):
    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
        persona = init_persona
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        convo_str = ""
        for i in curr_chat:
            convo_str += ": ".join(i) + "\n"
        if convo_str == "":
            convo_str = "[The conversation has not started yet -- start it!]"

        personal_information = init_persona.scratch.get_str_iss()
        relationship = init_persona.scratch.get_relationship_feeling(target_persona.scratch.name)
        personal_information += f"Relationship with {target_persona.scratch.name}: {relationship['relationship']}\n"
        personal_information += f"Feeling about {target_persona.scratch.name}: {relationship['feeling']}\n"

        # init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
        prompt_input = [personal_information, init_persona.scratch.name, retrieved, prev_convo_insert,
                        curr_location, curr_context, target_persona.scratch.name,
                        convo_str]
        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = json.loads(gpt_response)

        cleaned_dict = dict()
        cleaned = []
        for key, val in gpt_response.items():
            cleaned += [val]
        cleaned_dict["utterance"] = cleaned[0]
        cleaned_dict["end"] = True
        if "f" in str(cleaned[1]) or "F" in str(cleaned[1]):
            cleaned_dict["end"] = False

        return cleaned_dict

    def __chat_func_validate(gpt_response, prompt=""):
        print("ugh...")
        try:
            # print ("debug 1")
            # print (gpt_response)
            # print ("debug 2")
            print(json.loads(gpt_response))
            # print ("debug 3")

            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = dict()
        cleaned_dict["utterance"] = "..."
        cleaned_dict["end"] = False
        return cleaned_dict

    print("11")
    prompt_template = "persona/prompt_template/lifestyle/iterative_convo.txt"
    prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    gpt_param = {"engine": "gpt-3.5-turbo-instruct", "api_type": "openai", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_generate_iterative_chat_utt_origin(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                        test_input=None, verbose=False):
    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
        persona = init_persona
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        retrieved_str = ""
        for key, vals in retrieved.items():
            for v in vals:
                retrieved_str += f"- {v.description}\n"

        convo_str = ""
        for i in curr_chat:
            convo_str += ": ".join(i) + "\n"
        if convo_str == "":
            convo_str = "[The conversation has not started yet -- start it!]"

        init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
        prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
                        curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
                        convo_str, init_persona.scratch.name, target_persona.scratch.name,
                        init_persona.scratch.name, init_persona.scratch.name,
                        init_persona.scratch.name
                        ]
        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = extract_first_json_dict(gpt_response)

        cleaned_dict = dict()
        cleaned = []
        for key, val in gpt_response.items():
            cleaned += [val]
        cleaned_dict["utterance"] = cleaned[0]
        cleaned_dict["end"] = True
        if "f" in str(cleaned[1]) or "F" in str(cleaned[1]):
            cleaned_dict["end"] = False

        return cleaned_dict

    def __chat_func_validate(gpt_response, prompt=""):
        print("ugh...")
        try:
            # print ("debug 1")
            # print (gpt_response)
            # print ("debug 2")

            print(extract_first_json_dict(gpt_response))
            # print ("debug 3")

            return True
        except Exception as e:
            metrics.fail_record(e)
            return False

    def get_fail_safe():
        cleaned_dict = dict()
        cleaned_dict["utterance"] = "..."
        cleaned_dict["end"] = False
        return cleaned_dict

    print("11")
    prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt"
    prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    gpt_param = {"engine": "text-davinci-003", "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_generate_retrieved_summary(name, retrieved):
    # get prompt input
    prompt_input = f"The following sentences are thoughts in {name}'s Head, " \
                   "please summary them in a brief and precise statement:\n"
    key_description = ""

    for v in retrieved:
        key_description += f"- {v.description}\n"

    print('retrieved_summary')
    print(f"Input:\n {prompt_input + key_description}")
    try:
        output = ChatGPT_single_request(prompt_input + key_description)
        print(f"Output:\n {output}")
        return output
    except Exception as e:
        metrics.fail_record(e)
    return key_description


def run_gpt_update_retrieved_summary(name, summary, retrieved):
    key_description = ""
    for v in retrieved:
        key_description += f"- {v.description}\n"

    format = '{"summary":"..."}'

    prompt_input = f"This is the summary of thoughts in {name}'s Head:\n" \
                   f"{summary}\n\n" \
                   f"The following sentence are new thoughts in {name}'s Head:\n" \
                   f"{key_description}\n" \
                   f"Considering the summary of the preceding thoughts, " \
                   f"do these new statements contribute any additional information?\n" \
                   f"If yes, please update the summary in a brief and precise way else just repeat the summary.\n\n" \
                   f"Please response in the JSON format WITHOUT any extra statement:\n{format}"

    print('update_retrieved_summary')
    print(f"Input:\n {prompt_input}")
    try:
        output = ChatGPT_single_request(prompt_input)
        print(f"Output:\n {output}")
        return json.loads(output)['summary']
    except Exception as e:
        metrics.fail_record(e)
    return key_description


def run_gpt_update_relationship(init_persona, target_persona, conversation_summary, summary):
    self_information = init_persona.scratch.get_str_iss()

    relationship = init_persona.scratch.get_relationship_feeling(target_persona.scratch.name)

    format = """
    {
        "relationship": "...",
        "feeling": "..."
    }
    """

    example = """
    Here is a example response:
    {
        "relationship": "close friend",
        "feeling": "friendly and collaborative"
    }
    """
    init_curr_index = init_persona.scratch.get_f_daily_schedule_index()
    init_act_desp, _ = init_persona.scratch.f_daily_schedule[init_curr_index]

    target_curr_index = target_persona.scratch.get_f_daily_schedule_index()
    target_act_desp, _ = target_persona.scratch.f_daily_schedule[target_curr_index]

    prompt_input = f"Here is the personal information about {init_persona.scratch.get_str_firstname()}:\n" \
                   f"{self_information}\n\n" \
                   f"{init_persona.scratch.get_str_firstname()}'s activity before chat: {init_act_desp}\n\n" \
                   f"{target_persona.scratch.get_str_firstname()}'s activity before chat: {target_act_desp}\n\n" \
                   f"Here is the current thoughts in {init_persona.scratch.get_str_firstname()}'s head:\n" \
                   f"{summary}\n\n" \
                   f"Here is the conversation summary between {init_persona.scratch.get_str_firstname()} and {target_persona.scratch.get_str_firstname()}:\n" \
                   f"{conversation_summary}\n\n" \
                   f"Current relationship between {init_persona.scratch.get_str_firstname()} and {target_persona.scratch.get_str_firstname()}:" \
                   f"{relationship['relationship']}\n" \
                   f"Current {init_persona.scratch.get_str_firstname()}'s feeling about {target_persona.scratch.get_str_firstname()}:" \
                   f"{relationship['feeling']}\n\n" \
                   f"From about information, do you need to update the relationship between " \
                   f"{init_persona.scratch.get_str_firstname()} and {target_persona.scratch.get_str_firstname()}." \
                   f" And how {init_persona.scratch.get_str_firstname()} feel about {target_persona.scratch.get_str_firstname()}?\n" \
                   f"Please ONLY response in the JSON format WITHOUT any extra statement:\n" \
                   f"{format}\n" \
                   f"{example}"

    print('update_relationship')
    print(f"Input:\n {prompt_input}")
    try:
        output = ChatGPT_single_request(prompt_input)
        print(f"Output:\n {output}")
        json_data = json.loads(output)
        assert 'relationship' in json_data.keys()
        assert 'feeling' in json_data.keys()
        return json_data
    except Exception as e:
        metrics.fail_record(e)
    return {
        "relationship": "unknown",
        "feeling": "unknown",
    }
