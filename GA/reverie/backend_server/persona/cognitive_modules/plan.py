"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: plan.py
Description: This defines the "Plan" module for generative agents. 
"""
import datetime
import math
import random
import sys
import time
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.run_gpt_prompt import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.converse import *

from pool import get_policy_pool, get_relate_policy, update_policy_pool, get_sub_task_pool, update_sub_task_pool

from record import update_record_tree


##############################################################################
# CHAPTER 2: Generate
##############################################################################

def generate_wake_up_hour(persona):
    """
  Generates the time when the persona wakes up. This becomes an integral part
  of our process for generating the persona's daily plan.

  Persona state: identity stable set, lifestyle, first_name

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    an integer signifying the persona's wake up hour
  EXAMPLE OUTPUT:
    8
  """
    if debug: print("GNS FUNCTION: <generate_wake_up_hour>")
    return int(run_gpt_prompt_wake_up_hour(persona)[0])


def generate_first_daily_plan(persona, wake_up_hour):
    """
  Generates the daily plan for the persona.
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form:
  'wake up and complete the morning routine at 6:00 am',
  'eat breakfast at 7:00 am',..
  Note that the actions come without a period.

  Persona state: identity stable set, lifestyle, cur_data_str, first_name

  INPUT:
    persona: The Persona class instance
    wake_up_hour: an integer that indicates when the hour the persona wakes up
                  (e.g., 8)
  OUTPUT:
    a list of daily actions in broad strokes.
  EXAMPLE OUTPUT:
    ['wake up and complete the morning routine at 6:00 am',
     'have breakfast and brush teeth at 6:30 am',
     'work on painting project from 8:00 am to 12:00 pm',
     'have lunch at 12:00 pm',
     'take a break and watch TV from 2:00 pm to 4:00 pm',
     'work on painting project from 4:00 pm to 6:00 pm',
     'have dinner at 6:00 pm', 'watch TV from 7:00 pm to 8:00 pm']
  """
    if debug: print("GNS FUNCTION: <generate_first_daily_plan>")

    return run_gpt_prompt_daily_plan(persona, wake_up_hour)[0]


def generate_first_daily_plan_with_extra_idea(persona, wake_up_hour):
    if debug: print("GNS FUNCTION: <generate_first_daily_plan_with_extra_idea>")

    extra_plan = run_gpt_extra_plan(persona)[0]

    return run_gpt_prompt_daily_plan_with_extra_plan(persona, extra_plan, wake_up_hour)[0]


def generate_hourly_schedule(daily_plan, wake_up_hour, curr_time):
    daily_req = []
    n_m1_hourly_compressed = []

    curr_min = int(curr_time.hour * 60 + curr_time.minute)

    rest_time = 24 * 60
    n_m1_hourly_compressed += [["sleeping", int(wake_up_hour * 60) - curr_min]]
    rest_time -= int(wake_up_hour * 60) - curr_min

    for activity_json in daily_plan:
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

    if rest_time > 0:
        n_m1_hourly_compressed += [["sleeping", int(rest_time)]]

    assert len(n_m1_hourly_compressed) >= 7, "generate_first_daily_plan -> activities num should larger than 7"

    print(f"generate_hourly_schedule -> activity: {n_m1_hourly_compressed}")
    return daily_req, n_m1_hourly_compressed


def generate_task_decomp(persona, task, duration):
    """
  A few shot decomposition of a task given the task description

  Persona state: identity stable set, curr_date_str, first_name

  INPUT:
    persona: The Persona class instance
    task: the description of the task at hand in str form
          (e.g., "waking up and starting her morning routine")
    duration: an integer that indicates the number of minutes this task is
              meant to last (e.g., 60)
  OUTPUT:
    a list of list where the inner list contains the decomposed task
    description and the number of minutes the task is supposed to last.
  EXAMPLE OUTPUT:
    [['going to the bathroom', 5], ['getting dressed', 5],
     ['eating breakfast', 15], ['checking her email', 5],
     ['getting her supplies ready for the day', 15],
     ['starting to work on her painting', 15]]

  """
    if debug: print("GNS FUNCTION: <generate_task_decomp>")

    exist_policy = get_policy_pool(persona.name, task, duration)
    if exist_policy:
        return exist_policy, False, task

    embedding = get_embedding(task)
    exist_policy, task_name = get_relate_policy(persona.name, task, duration, embedding)
    if exist_policy:
        return exist_policy, False, task_name

    policy = run_gpt_prompt_task_decomp(persona, task, int(duration))[0]
    update_policy_pool(persona.name, task, duration, policy, embedding)
    return policy, True, task

def generate_task_decomp_with_extra_mind(persona, task, duration):
    """
  A few shot decomposition of a task given the task description and extra mind
  """
    if debug: print("GNS FUNCTION: <generate_task_decomp_with_extra_mind>")

    extra_mind = get_addition_information(persona, 5)

    extra_mind_str = ""
    for i, v in enumerate(extra_mind):
        extra_mind_str += f"{i + 1}. {v}\n"

    policy = run_gpt_prompt_task_decomp_with_extra_mind(persona, task, int(duration), extra_mind_str)[0]
    embedding = get_embedding(task)
    # exist same policy name
    exist_policy = get_policy_pool(persona.name, task, duration)
    if exist_policy:
        if check_same_policy(exist_policy, policy):
            return exist_policy, False, task
    # similar policy name
    exist_policy, task_name = get_relate_policy(persona.name, task, duration, embedding)
    if exist_policy:
        if check_same_policy(exist_policy, policy):
            return exist_policy, False, task

    update_policy_pool(persona.name, task, duration, policy, embedding)
    return policy, True, task


def check_same_policy(policy1, policy2):
    if len(policy1) != len(policy2):
        return False

    for p1, p2 in zip(policy1, policy2):
        activity1, activity2 = p1['activity'], p2['activity']
        duration1, duration2 = p1['duration'], p2['duration']
        if duration1 != duration2:
            return False
        if cos_sim(get_embedding(activity1), get_embedding(activity2)) < 0.97:
            return False
    return True

def generate_action_sector(act_desp, persona, maze):
    """TODO
  Given the persona and the task description, choose the action_sector.

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT:
    act_desp: description of the new action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT:
    action_arena (e.g., "bedroom 2")
  EXAMPLE OUTPUT:
    "bedroom 2"
  """
    if debug: print("GNS FUNCTION: <generate_action_sector>")
    return run_gpt_prompt_action_sector(act_desp, persona, maze)[0]


def generate_action_arena(act_desp, persona, maze, act_world, act_sector):
    """TODO
  Given the persona and the task description, choose the action_arena.

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT:
    act_desp: description of the new action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT:
    action_arena (e.g., "bedroom 2")
  EXAMPLE OUTPUT:
    "bedroom 2"
  """
    if debug: print("GNS FUNCTION: <generate_action_arena>")
    return run_gpt_prompt_action_arena(act_desp, persona, maze, act_world, act_sector)[0]


def generate_action_game_object(act_desp, act_address, persona, maze):
    """TODO
  Given the action description and the act address (the address where
  we expect the action to task place), choose one of the game objects.

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT:
    act_desp: the description of the action (e.g., "sleeping")
    act_address: the arena where the action will take place:
               (e.g., "dolores double studio:double studio:bedroom 2")
    persona: The Persona class instance
  OUTPUT:
    act_game_object:
  EXAMPLE OUTPUT:
    "bed"
  """
    if debug: print("GNS FUNCTION: <generate_action_game_object>")
    if not persona.s_mem.get_str_accessible_arena_game_objects(act_address):
        return "<random>"
    return run_gpt_prompt_action_game_object(act_desp, persona, maze, act_address)[0]


def generate_action_pronunciatio(act_desp, persona):
    """TODO
  Given an action description, creates an emoji string description via a few
  shot prompt.

  Does not really need any information from persona.

  INPUT:
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT:
    a string of emoji that translates action description.
  EXAMPLE OUTPUT:
    "🧈🍞"
  """
    if debug: print("GNS FUNCTION: <generate_action_pronunciatio>")
    try:
        x = run_gpt_prompt_pronunciatio(act_desp, persona)[0]
    except Exception as e:
        metrics.fail_record(e)
        x = "🙂"

    if not x:
        return "🙂"
    return x


def generate_action_event_triple(act_desp, persona):
    """TODO

  INPUT:
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT:
    a string of emoji that translates action description.
  EXAMPLE OUTPUT:
    "🧈🍞"
  """
    if debug: print("GNS FUNCTION: <generate_action_event_triple>")
    return run_gpt_prompt_event_triple(act_desp, persona)[0]


def generate_act_obj_desc(act_game_object, act_desp, persona):
    if debug: print("GNS FUNCTION: <generate_act_obj_desc>")
    return run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona)[0]


def generate_act_obj_event_triple(act_game_object, act_obj_desc, persona):
    if debug: print("GNS FUNCTION: <generate_act_obj_event_triple>")
    return run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona)[0]


def generate_convo(maze, init_persona, target_persona):
    curr_loc = maze.access_tile(init_persona.scratch.curr_tile)

    # convo = run_gpt_prompt_create_conversation(init_persona, target_persona, curr_loc)[0]
    # convo = agent_chat_v1(maze, init_persona, target_persona)
    convo, target_summary, init_summary = agent_chat_v2(maze, init_persona, target_persona)
    all_utt = ""

    init_persona.scratch.relationship[target_persona.scratch.name] = \
        run_gpt_update_relationship(init_persona, target_persona, convo, target_summary)
    target_persona.scratch.relationship[init_persona.scratch.name] = \
        run_gpt_update_relationship(target_persona, init_persona, convo, init_summary)

    for row in convo:
        speaker = row[0]
        utt = row[1]
        all_utt += f"{speaker}: {utt}\n"

    convo_length = math.ceil(int(len(all_utt) / 8) / 30)

    if debug: print("GNS FUNCTION: <generate_convo>")
    return convo, convo_length


def generate_convo_summary(persona, convo):
    convo_summary = run_gpt_prompt_summarize_conversation(persona, convo)[0]
    return convo_summary


# TODO generate_decide_to_talk refine the answer make agent not always say yes
def generate_decide_to_talk(init_persona, target_persona, retrieved):
    x = run_gpt_prompt_decide_to_talk(init_persona, target_persona, retrieved)[0]
    if debug: print("GNS FUNCTION: <generate_decide_to_talk>")

    if x == "yes":
        return True
    else:
        return False


# TODO refine generate_decide_to_react
def generate_decide_to_react(init_persona, target_persona, retrieved):
    if debug: print("GNS FUNCTION: <generate_decide_to_react>")
    return run_gpt_prompt_decide_to_react(init_persona, target_persona, retrieved)[0]


def get_split_minute_schedule(minute_schedule):
    if "(" in minute_schedule:
        hourly_schedule = minute_schedule.split("(")[0].strip()
        rest_schedule = minute_schedule.split("(")[1].replace(")", "")
        return hourly_schedule, rest_schedule
    return minute_schedule, minute_schedule


def generate_new_decomp_schedule(persona, inserted_act, inserted_act_dur, start_hour, end_hour):
    # Step 1: Setting up the core variables for the function.
    # <p> is the persona whose schedule we are editing right now.
    p = persona
    # <today_min_pass> indicates the number of minutes that have passed today.
    today_min_pass = (int(p.scratch.curr_time.hour) * 60
                      + int(p.scratch.curr_time.minute) + 1)

    # Step 2: We need to create <main_act_dur> and <truncated_act_dur>.
    # These are basically a sub-component of <f_daily_schedule> of the persona,
    # but focusing on the current decomposition.
    # Here is an example for <main_act_dur>:
    # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
    # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
    # ['wakes up and completes her morning routine (uses the restroom)', 5]
    # ['wakes up and completes her morning routine (washes her ...)', 10]
    # ['wakes up and completes her morning routine (makes her bed)', 5]
    # ['wakes up and completes her morning routine (eats breakfast)', 15]
    # ['wakes up and completes her morning routine (gets dressed)', 10]
    # ['wakes up and completes her morning routine (leaves her ...)', 5]
    # ['wakes up and completes her morning routine (starts her ...)', 5]
    # ['preparing for her day (waking up at 6am)', 5]
    # ['preparing for her day (making her bed)', 5]
    # ['preparing for her day (taking a shower)', 15]
    # ['preparing for her day (getting dressed)', 5]
    # ['preparing for her day (eating breakfast)', 10]
    # ['preparing for her day (brushing her teeth)', 5]
    # ['preparing for her day (making coffee)', 5]
    # ['preparing for her day (checking her email)', 5]
    # ['preparing for her day (starting to work on her painting)', 5]
    #
    # And <truncated_act_dur> concerns only until where an event happens.
    # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
    # ['wakes up and completes her morning routine (wakes up at 6am)', 2]
    main_act_dur = []
    truncated_act_dur = []
    dur_sum = 0  # duration sum
    count = 0  # enumerate count
    truncated_fin = False
    hourly_schedule = None

    print("DEBUG::: ", persona.scratch.name)
    for act, dur in p.scratch.f_daily_schedule:
        if (dur_sum >= start_hour * 60) and (dur_sum < end_hour * 60):
            hourly_schedule, minute_schedule = get_split_minute_schedule(p.scratch.f_daily_schedule[count][0])
            main_act_dur += [[minute_schedule, dur]]
            if dur_sum <= today_min_pass:
                _, minute_schedule = get_split_minute_schedule(p.scratch.f_daily_schedule[count][0])
                truncated_act_dur += [[minute_schedule, dur]]
            elif dur_sum > today_min_pass and not truncated_fin:
                # We need to insert that last act, duration list like this one:
                # e.g., ['wakes up and completes her morning routine (wakes up...)', 2]
                _, minute_schedule = get_split_minute_schedule(p.scratch.f_daily_schedule[count][0])
                truncated_act_dur += [[minute_schedule,
                                       dur_sum - today_min_pass]]
                truncated_act_dur[-1][-1] -= (
                        dur_sum - today_min_pass)  ######## DEC 7 DEBUG;.. is the +1 the right thing to do???
                # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass + 1) ######## DEC 7 DEBUG;.. is the +1 the right thing to do???
                print("DEBUG::: ", truncated_act_dur)

                # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do???
                truncated_fin = True
        dur_sum += dur
        count += 1

    persona_name = persona.name
    main_act_dur = main_act_dur

    # x = truncated_act_dur[-1][0].split("(")[0].strip() + " (on the way to " + truncated_act_dur[-1][0].split("(")[-1][
    #                                                                           :-1] + ")"
    # truncated_act_dur[-1][0] = x

    # if "(" in truncated_act_dur[-1][0]:
    #     inserted_act = truncated_act_dur[-1][0].split("(")[0].strip() + " (" + inserted_act + ")"

    # To do inserted_act_dur+1 below is an important decision but I'm not sure
    # if I understand the full extent of its implications. Might want to
    # revisit.
    truncated_act_dur += [[inserted_act, inserted_act_dur]]
    start_time_hour = (datetime.datetime(2022, 10, 31, 0, 0)
                       + datetime.timedelta(hours=start_hour))
    end_time_hour = (datetime.datetime(2022, 10, 31, 0, 0)
                     + datetime.timedelta(hours=end_hour))

    if debug: print("GNS FUNCTION: <generate_new_decomp_schedule>")

    json_data = run_gpt_prompt_new_decomp_schedule(persona,
                                                   hourly_schedule,
                                                   main_act_dur,
                                                   truncated_act_dur,
                                                   start_time_hour,
                                                   end_time_hour,
                                                   inserted_act,
                                                   inserted_act_dur)[0]

    ret = []
    for single_activity in json_data:
        activity = hourly_schedule + "(" + single_activity['activity'] + ")"
        start_time = datetime.datetime.strptime(single_activity["start"], '%H:%M %p')
        end_time = datetime.datetime.strptime(single_activity["end"], '%H:%M %p')
        min_diff_time = (end_time - start_time).total_seconds() / 60
        if min_diff_time < 0:
            min_diff_time = 1440 + min_diff_time
        ret.append([activity, int(min_diff_time)])

    return ret


##############################################################################
# CHAPTER 3: Plan
##############################################################################

def revise_identity(persona, wake_up_hour):
    p_name = persona.scratch.name

    focal_points = [f"{p_name}'s plan for {persona.scratch.get_str_curr_date_str()}.",
                    f"Important recent events for {p_name}'s life."]
    retrieved = new_retrieve(persona, focal_points, 5)

    statements = ""
    for key, val in retrieved.items():
        for i in val:
            statements += f"{i.created.strftime('%a %b %d-%H:%M %p')}: {i.embedding_key}\n"

    plan_note_list = run_gpt_prompt_next_day_remember(persona, statements)[0]
    print(f"plan_note_list -> {plan_note_list}")
    plan_note = ""
    for index, single_plan_note in enumerate(plan_note_list):
        plan_note += f"{index + 1}.{single_plan_note['remind']}\n"
    if plan_note == "":
        plan_note = "None\n"

    print(f"plan_note -> {plan_note}")

    thought_prompt = statements + "\n"
    thought_prompt += f"Given the statements above, how might we summarize {p_name}'s feelings about their days up to now?\n\n"
    thought_prompt += f"Write the response from {p_name}'s perspective."
    thought_note = ChatGPT_single_request(thought_prompt)
    print(f"thought_note -> {thought_note}")

    if True:
        addition_information = get_addition_information(persona, 3)
        extra_plan_list = run_gpt_next_day_extra_plan(persona, addition_information)[0]
        extra_plan_str = ""
        for i, v in enumerate(extra_plan_list):
            extra_plan_str += f"{i+1}. {v['output']}"
        new_daily_req = \
            run_gpt_prompt_next_day_plan_with_extra_plan(persona, wake_up_hour, plan_note, thought_note, extra_plan_str)[0]
    else:
        new_daily_req = run_gpt_prompt_next_day_plan(persona, wake_up_hour, plan_note, thought_note)[0]

    print("WE ARE HERE!!!", new_daily_req)
    return new_daily_req


def get_addition_information(persona, num):
    embeddings = persona.a_mem.embeddings
    nodes_dict = persona.a_mem.id_to_node

    key_list = []
    embedding_list = []
    for k, v in embeddings.items():
        key_list.append(k)
        embedding_list.append(v)

    embeddings = np.array(embedding_list)

    return sample_dict_with_weight(key_list, embeddings, num)


def sample_dict_with_weight(key_list, embeddings, num):
    dbscan = DBSCAN(eps=0.3, min_samples=2)
    dbscan.fit(embeddings)

    # 获取簇标签
    labels = dbscan.labels_

    clusters = set(labels) - {-1}

    # 创建一个字典，键是簇标签，值是对应簇的大小
    cluster_sizes = {cluster: np.sum(labels == cluster) for cluster in clusters}

    probabilities = []
    for key, label in zip(key_list, labels):
        if label in cluster_sizes.keys():
            probabilities.append(1 / (cluster_sizes[label]))
        else:
            probabilities.append(1)

    all_probabilities_value = sum(probabilities)
    probabilities = [p / all_probabilities_value for p in probabilities]

    sampled_names = np.random.choice(key_list, size=num, replace=False, p=probabilities)

    return sampled_names


def density_score(max_local_density, density):
    ret = density / max_local_density * 10
    return 10 - ret


def _long_term_planning(persona, new_day):
    """
  Formulates the persona's daily long-term plan if it is the start of a new
  day. This basically has two components: first, we create the wake-up hour,
  and second, we create the hourly schedule based on it.
  INPUT
    new_day: Indicates whether the current time signals a "First day",
             "New day", or False (for neither). This is important because we
             create the personas' long term planning on the new day.
  """
    # We start by creating the wake up hour for the persona.
    wake_up_hour = generate_wake_up_hour(persona)
    curr_time = persona.scratch.curr_time
    # When it is a new day, we start by creating the daily_req of the persona.
    # Note that the daily_req is a list of strings that describe the persona's
    # day in broad strokes.
    if new_day == "First day":
        # Bootstrapping the daily plan for the start of then generation:
        # if this is the start of generation (so there is no previous day's
        # daily requirement, or if we are on a new day, we want to create a new
        # set of daily requirements.
        if True:
            daily_plan = generate_first_daily_plan_with_extra_idea(persona, wake_up_hour)
        else:
            daily_plan = generate_first_daily_plan(persona, wake_up_hour)
        daily_req, compressed_activity = generate_hourly_schedule(daily_plan, wake_up_hour, curr_time)

        persona.scratch.daily_req = daily_req
        persona.scratch.f_daily_schedule = compressed_activity
    elif new_day == "New day":
        daily_plan = revise_identity(persona, wake_up_hour)
        daily_req, compressed_activity = generate_hourly_schedule(daily_plan, wake_up_hour, curr_time)
        persona.scratch.daily_req = daily_req
        persona.scratch.f_daily_schedule = compressed_activity

    # Based on the daily_req, we create an hourly schedule for the persona,
    # which is a list of todo items with a time duration (in minutes) that
    # add up to 24 hours.
    # persona.scratch.f_daily_schedule = generate_hourly_schedule(persona,
    #                                                             wake_up_hour)
    persona.scratch.f_daily_new_plan = [False] * len(persona.scratch.f_daily_schedule)
    persona.scratch.f_daily_parent_plan = ['root'] * len(persona.scratch.f_daily_schedule)
    persona.scratch.f_daily_schedule_hourly_org = (persona.scratch
                                                       .f_daily_schedule[:])

    # Added March 4 -- adding plan to the memory.
    thought = f"This is {persona.scratch.name}'s plan for {persona.scratch.curr_time.strftime('%a %b %d')}:"
    for i in persona.scratch.daily_req:
        thought += f" {i},"
    thought = thought[:-1] + "."
    created = persona.scratch.curr_time
    expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
    s, p, o = (persona.scratch.name, "plan", persona.scratch.curr_time.strftime('%a %b %d'))
    keywords = set(["plan"])
    thought_poignancy = 5
    thought_embedding_pair = (thought, get_embedding(thought))
    persona.a_mem.add_thought(created, expiration, s, p, o,
                              thought, keywords, thought_poignancy,
                              thought_embedding_pair, None)

    # print("Sleeping for 20 seconds...")
    # time.sleep(10)
    # print("Done sleeping!")


def _determine_action(persona, maze):
    """
  Creates the next action sequence for the persona.
  The main goal of this function is to run "add_new_action" on the persona's
  scratch space, which sets up all the action related variables for the next
  action.
  As a part of this, the persona may need to decompose its hourly schedule as
  needed.
  INPUT
    persona: Current <Persona> instance whose action we are determining.
    maze: Current <Maze> instance.
  """

    def determine_decomp(act_desp, act_dura):
        """
    Given an action description and its duration, we determine whether we need
    to decompose it. If the action is about the agent sleeping, we generally
    do not want to decompose it, so that's what we catch here.

    INPUT:
      act_desp: the description of the action (e.g., "sleeping")
      act_dura: the duration of the action in minutes.
    OUTPUT:
      a boolean. True if we need to decompose, False otherwise.
    """
        if "sleep" not in act_desp and "bed" not in act_desp:
            return True
        elif "sleeping" in act_desp or "asleep" in act_desp or "in bed" in act_desp:
            return False
        elif "sleep" in act_desp or "bed" in act_desp:
            if act_dura > 60:
                return False
        return True

    # The goal of this function is to get us the action associated with
    # <curr_index>. As a part of this, we may need to decompose some large
    # chunk actions.
    # Importantly, we try to decompose at least two hours worth of schedule at
    # any given point.
    curr_index = persona.scratch.get_f_daily_schedule_index()
    curr_index_60 = persona.scratch.get_f_daily_schedule_index(advance=60)

    # * Decompose *
    # During the first hour of the day, we need to decompose two hours
    # sequence. We do that here.
    if curr_index == 0:
        # This portion is invoked if it is the first hour of the day.
        act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
        if act_dura >= 60:
            # We decompose if the next action is longer than an hour, and fits the
            # criteria described in determine_decomp.
            if determine_decomp(act_desp, act_dura):
                sub_task, is_new_plan, task_name = generate_task_decomp(persona, act_desp, act_dura)
                persona.scratch.f_daily_schedule[curr_index:curr_index + 1] = sub_task
                persona.scratch.f_daily_new_plan[curr_index:curr_index + 1] = [is_new_plan] * len(sub_task)
                persona.scratch.f_daily_parent_plan[curr_index:curr_index + 1] = [task_name] * len(sub_task)
                for _sub_task in sub_task:
                    persona.scratch.f_daily_subtask_parent[_sub_task[0]] = act_desp

        if curr_index_60 + 1 < len(persona.scratch.f_daily_schedule):
            act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60 + 1]
            if act_dura >= 60:
                if determine_decomp(act_desp, act_dura):
                    sub_task, is_new_plan, task_name = generate_task_decomp(persona, act_desp, act_dura)
                    persona.scratch.f_daily_schedule[curr_index_60 + 1:curr_index_60 + 2] = sub_task
                    persona.scratch.f_daily_new_plan[curr_index_60 + 1:curr_index_60 + 2] = [is_new_plan] * len(
                        sub_task)
                    persona.scratch.f_daily_parent_plan[curr_index_60 + 1:curr_index_60 + 2] = [task_name] * len(
                        sub_task)
                    for _sub_task in sub_task:
                        persona.scratch.f_daily_subtask_parent[_sub_task[0]] = act_desp

    if curr_index_60 < len(persona.scratch.f_daily_schedule):
        # If it is not the first hour of the day, this is always invoked (it is
        # also invoked during the first hour of the day -- to double up so we can
        # decompose two hours in one go). Of course, we need to have something to
        # decompose as well, so we check for that too.
        if persona.scratch.curr_time.hour < 23:
            # And we don't want to decompose after 11 pm.
            act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60]
            if act_dura >= 60:
                if determine_decomp(act_desp, act_dura):
                    sub_task, is_new_plan, task_name = generate_task_decomp(persona, act_desp, act_dura)
                    persona.scratch.f_daily_schedule[curr_index_60:curr_index_60 + 1] = sub_task
                    persona.scratch.f_daily_new_plan[curr_index_60:curr_index_60 + 1] = [is_new_plan] * len(
                        sub_task)
                    persona.scratch.f_daily_parent_plan[curr_index_60:curr_index_60 + 1] = [task_name] * len(
                        sub_task)
                    for _sub_task in sub_task:
                        persona.scratch.f_daily_subtask_parent[_sub_task[0]] = act_desp
    # * End of Decompose *

    # Generate an <Action> instance from the action description and duration. By
    # this point, we assume that all the relevant actions are decomposed and
    # ready in f_daily_schedule.
    print("DEBUG LJSDLFSKJF")
    for i in persona.scratch.f_daily_schedule: print(i)
    print(curr_index)
    print(len(persona.scratch.f_daily_schedule))
    print(persona.scratch.name)
    print("------")

    # 1440
    x_emergency = 0
    for i in persona.scratch.f_daily_schedule:
        x_emergency += i[1]
    # print ("x_emergency", x_emergency)

    if 1440 - x_emergency > 0:
        print("x_emergency__AAA", x_emergency)
    persona.scratch.f_daily_schedule += [["sleeping", 1440 - x_emergency]]
    persona.scratch.f_daily_new_plan += [False]
    persona.scratch.f_daily_parent_plan += ['root']

    act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
    new_plan_emoji = persona.scratch.f_daily_new_plan[curr_index]
    parent_plan = persona.scratch.f_daily_parent_plan[curr_index]
    pre_emoji = "🧠🚫->"
    if new_plan_emoji:
        pre_emoji = "🧠💡->"
    else:
        if parent_plan == 'root' and act_dura < 60 and act_desp != 'sleeping':
            task_info = get_policy_pool(persona.name, act_desp, act_dura)
            if task_info is None:
                task_embedding = get_embedding(act_desp)
                _, task_name = get_relate_policy(persona.name, act_desp, act_dura, task_embedding)
                if task_name is None:
                    pre_emoji = "🧠💡->"
                    persona.scratch.f_daily_new_plan[curr_index] = True
                    new_plan_emoji = True
                    update_policy_pool(persona.name, act_desp, act_dura, [], task_embedding)
                else:
                    act_desp = task_name

    update_record_tree(persona.name, act_desp, parent_plan, new_plan_emoji, act_dura, persona.scratch.curr_time, {})

    exist_sub_task = get_sub_task_pool(persona.name, act_desp)
    if exist_sub_task is not None:
        new_address = exist_sub_task["new_address"]
        act_pron = exist_sub_task["act_pron"]
        act_event = exist_sub_task["act_event"]
        act_game_object = exist_sub_task["act_game_object"]
        act_obj_desp = exist_sub_task["act_obj_desp"]
        act_obj_pron = exist_sub_task["act_obj_pron"]
        act_obj_event = exist_sub_task["act_obj_event"]
    else:
        # Finding the target location of the action and creating action-related
        # variables.
        act_world = maze.access_tile(persona.scratch.curr_tile)["world"]
        # act_sector = maze.access_tile(persona.scratch.curr_tile)["sector"]
        act_sector = generate_action_sector(act_desp, persona, maze)
        act_arena = generate_action_arena(act_desp, persona, maze, act_world, act_sector)
        act_address = f"{act_world}:{act_sector}:{act_arena}"
        act_game_object = generate_action_game_object(act_desp, act_address,
                                                      persona, maze)
        new_address = f"{act_world}:{act_sector}:{act_arena}:{act_game_object}"
        act_pron = generate_action_pronunciatio(act_desp, persona)
        act_event = generate_action_event_triple(act_desp, persona)
        # Persona's actions also influence the object states. We set those up here.
        act_obj_desp = generate_act_obj_desc(act_game_object, act_desp, persona)
        act_obj_pron = generate_action_pronunciatio(act_obj_desp, persona)
        act_obj_event = generate_act_obj_event_triple(act_game_object,
                                                      act_obj_desp, persona)

        sub_task_info = {
            "new_address": new_address,
            "act_pron": act_pron,
            "act_event": act_event,
            "act_game_object": act_game_object,
            "act_obj_desp": act_obj_desp,
            "act_obj_pron": act_obj_pron,
            "act_obj_event": act_obj_event,
        }

        update_sub_task_pool(persona.name, act_desp, sub_task_info)

    # Adding the action to persona's queue.
    persona.scratch.add_new_action(new_address,
                                   int(act_dura),
                                   act_desp,
                                   pre_emoji + act_pron,
                                   act_event,
                                   None,
                                   None,
                                   None,
                                   None,
                                   act_obj_desp,
                                   act_obj_pron,
                                   act_obj_event)


def _choose_retrieved(persona, retrieved):
    """
  Retrieved elements have multiple core "curr_events". We need to choose one
  event to which we are going to react to. We pick that event here.
  INPUT
    persona: Current <Persona> instance whose action we are determining.
    retrieved: A dictionary of <ConceptNode> that were retrieved from the
               the persona's associative memory. This dictionary takes the
               following form:
               dictionary[event.description] =
                 {["curr_event"] = <ConceptNode>,
                  ["events"] = [<ConceptNode>, ...],
                  ["thoughts"] = [<ConceptNode>, ...] }
  """
    # Once we are done with the reflection, we might want to build a more
    # complex structure here.

    # We do not want to take self events... for now
    copy_retrieved = retrieved.copy()
    for event_desc, rel_ctx in copy_retrieved.items():
        curr_event = rel_ctx["curr_event"]
        if curr_event.subject == persona.name:
            del retrieved[event_desc]

    # Always choose persona first.
    priority = []
    for event_desc, rel_ctx in retrieved.items():
        curr_event = rel_ctx["curr_event"]
        if (":" not in curr_event.subject
                and curr_event.subject != persona.name):
            priority += [rel_ctx]
    if priority:
        return random.choice(priority)

    # Skip idle.
    for event_desc, rel_ctx in retrieved.items():
        curr_event = rel_ctx["curr_event"]
        if "is idle" not in event_desc:
            priority += [rel_ctx]
    if priority:
        return random.choice(priority)
    return None


def _should_react(persona, retrieved, personas):
    """
  Determines what form of reaction the persona should exihibit given the
  retrieved values.
  INPUT
    persona: Current <Persona> instance whose action we are determining.
    retrieved: A dictionary of <ConceptNode> that were retrieved from the
               the persona's associative memory. This dictionary takes the
               following form:
               dictionary[event.description] =
                 {["curr_event"] = <ConceptNode>,
                  ["events"] = [<ConceptNode>, ...],
                  ["thoughts"] = [<ConceptNode>, ...] }
    personas: A dictionary that contains all persona names as keys, and the
              <Persona> instance as values.
  """

    def lets_talk(init_persona, target_persona, retrieved):
        if (not target_persona.scratch.act_address
                or not target_persona.scratch.act_description
                or not init_persona.scratch.act_address
                or not init_persona.scratch.act_description):
            return False

        if ("sleeping" in target_persona.scratch.act_description
                or "sleeping" in init_persona.scratch.act_description):
            return False

        if init_persona.scratch.curr_time.hour == 23:
            return False

        if "<waiting>" in target_persona.scratch.act_address:
            return False

        if (target_persona.scratch.chatting_with
                or init_persona.scratch.chatting_with):
            return False

        if (target_persona.name in init_persona.scratch.chatting_with_buffer):
            if init_persona.scratch.chatting_with_buffer[target_persona.name] > 0:
                return False

        if generate_decide_to_talk(init_persona, target_persona, retrieved):
            return True

        return False

    def lets_react(init_persona, target_persona, retrieved):
        if (not target_persona.scratch.act_address
                or not target_persona.scratch.act_description
                or not init_persona.scratch.act_address
                or not init_persona.scratch.act_description):
            return False

        if ("sleeping" in target_persona.scratch.act_description
                or "sleeping" in init_persona.scratch.act_description):
            return False

        # return False
        if init_persona.scratch.curr_time.hour == 23:
            return False

        if "waiting" in target_persona.scratch.act_description:
            return False
        if init_persona.scratch.planned_path == []:
            return False

        if (init_persona.scratch.act_address
                != target_persona.scratch.act_address):
            return False

        react_mode = generate_decide_to_react(init_persona,
                                              target_persona, retrieved)

        if react_mode == "1":
            wait_until = ((target_persona.scratch.act_start_time
                           + datetime.timedelta(minutes=target_persona.scratch.act_duration - 1))
                          .strftime("%B %d, %Y, %H:%M:%S"))
            return f"wait: {wait_until}"
        elif react_mode == "2":
            return False
            return "do other things"
        else:
            return False  # "keep"

    # If the persona is chatting right now, default to no reaction
    if persona.scratch.chatting_with:
        return False
    if "<waiting>" in persona.scratch.act_address:
        return False

    # Recall that retrieved takes the following form:
    # dictionary {["curr_event"] = <ConceptNode>,
    #             ["events"] = [<ConceptNode>, ...],
    #             ["thoughts"] = [<ConceptNode>, ...]}
    curr_event = retrieved["curr_event"]

    if ":" not in curr_event.subject:
        # this is a persona event.
        if lets_talk(persona, personas[curr_event.subject], retrieved):
            return f"chat with {curr_event.subject}"
        react_mode = lets_react(persona, personas[curr_event.subject],
                                retrieved)
        return react_mode
    return False


# TODO _create_react refine
def _create_react(persona, inserted_act, inserted_act_dur,
                  act_address, act_event, chatting_with, chat, chatting_with_buffer,
                  chatting_end_time,
                  act_pronunciatio, act_obj_description, act_obj_pronunciatio,
                  act_obj_event, act_start_time=None):
    p = persona

    min_sum = 0
    for i in range(p.scratch.get_f_daily_schedule_hourly_org_index()):
        min_sum += p.scratch.f_daily_schedule_hourly_org[i][1]
    start_hour = int(min_sum / 60)

    # if p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] >= 120:
    #     end_hour = start_hour + \
    #                p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] / 60
    #
    # elif p.scratch.get_f_daily_schedule_hourly_org_index() < len(p.scratch.f_daily_schedule_hourly_org) - 1:
    #     if (p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] +
    #             p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index() + 1][1]):
    #         end_hour = start_hour + (
    #                 (p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] +
    #                  p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index() + 1][
    #                      1]) / 60)
    # else:
    #     end_hour = start_hour + 2
    end_hour = start_hour + 1
    end_hour = int(end_hour)

    dur_sum = 0
    count = 0
    start_index = None
    end_index = None
    for act, dur in p.scratch.f_daily_schedule:
        if dur_sum >= start_hour * 60 and start_index == None:
            start_index = count
        if dur_sum >= end_hour * 60 and end_index == None:
            end_index = count
        dur_sum += dur
        count += 1

    ret = generate_new_decomp_schedule(p, inserted_act, inserted_act_dur,
                                       start_hour, end_hour)
    p.scratch.f_daily_schedule[start_index:end_index] = ret
    p.scratch.f_daily_new_plan[start_index:end_index] = [True] * len(ret)
    p.scratch.f_daily_parent_plan[start_index:end_index] = ['react'] * len(ret)
    p.scratch.add_new_action(act_address,
                             inserted_act_dur,
                             inserted_act,
                             act_pronunciatio,
                             act_event,
                             chatting_with,
                             chat,
                             chatting_with_buffer,
                             chatting_end_time,
                             act_obj_description,
                             act_obj_pronunciatio,
                             act_obj_event,
                             act_start_time)


def _chat_react(maze, persona, focused_event, reaction_mode, personas):
    # There are two personas -- the persona who is initiating the conversation
    # and the persona who is the target. We get the persona instances here.
    init_persona = persona
    target_persona = personas[reaction_mode[9:].strip()]
    curr_personas = [init_persona, target_persona]

    # Actually creating the conversation here.
    convo, duration_min = generate_convo(maze, init_persona, target_persona)
    convo_summary = generate_convo_summary(init_persona, convo)
    inserted_act = convo_summary
    inserted_act_dur = duration_min

    act_start_time = target_persona.scratch.act_start_time

    curr_time = target_persona.scratch.curr_time
    if curr_time.second != 0:
        temp_curr_time = curr_time + datetime.timedelta(seconds=60 - curr_time.second)
        chatting_end_time = temp_curr_time + datetime.timedelta(minutes=inserted_act_dur)
    else:
        chatting_end_time = curr_time + datetime.timedelta(minutes=inserted_act_dur)

    for role, p in [("init", init_persona), ("target", target_persona)]:
        if role == "init":
            act_address = f"<persona> {target_persona.name}"
            act_event = (p.name, "chat with", target_persona.name)
            chatting_with = target_persona.name
            chatting_with_buffer = {}
            chatting_with_buffer[target_persona.name] = 800
        elif role == "target":
            act_address = f"<persona> {init_persona.name}"
            act_event = (p.name, "chat with", init_persona.name)
            chatting_with = init_persona.name
            chatting_with_buffer = {}
            chatting_with_buffer[init_persona.name] = 800

        act_pronunciatio = "🧠💡->💬"
        act_obj_description = None
        act_obj_pronunciatio = None
        act_obj_event = (None, None, None)

        if role == "init":
            chat_info = {
                "person": target_persona.name,
                "message": convo
            }
            update_record_tree(persona.name, "chat", None, True, inserted_act_dur, persona.scratch.curr_time, chat_info)
        else:
            chat_info = {
                "person": persona.name,
                "message": convo
            }
            update_record_tree(target_persona.name, "chat", None, True, inserted_act_dur, persona.scratch.curr_time,
                               chat_info)

        _create_react(p, inserted_act, inserted_act_dur,
                      act_address, act_event, chatting_with, convo, chatting_with_buffer, chatting_end_time,
                      act_pronunciatio, act_obj_description, act_obj_pronunciatio,
                      act_obj_event, act_start_time)


def _wait_react(persona, reaction_mode):
    p = persona

    inserted_act = f'waiting to start {p.scratch.act_description.split("(")[-1][:-1]}'
    end_time = datetime.datetime.strptime(reaction_mode[6:].strip(), "%B %d, %Y, %H:%M:%S")
    inserted_act_dur = (end_time.minute + end_time.hour * 60) - (
            p.scratch.curr_time.minute + p.scratch.curr_time.hour * 60) + 1

    act_address = f"<waiting> {p.scratch.curr_tile[0]} {p.scratch.curr_tile[1]}"
    act_event = (p.name, "waiting to start", p.scratch.act_description.split("(")[-1][:-1])
    chatting_with = None
    chat = None
    chatting_with_buffer = None
    chatting_end_time = None

    act_pronunciatio = "🧠💡->⌛"
    act_obj_description = None
    act_obj_pronunciatio = None
    act_obj_event = (None, None, None)

    update_record_tree(persona.name, "react", None, True, inserted_act_dur, persona.scratch.curr_time, {})

    _create_react(p, inserted_act, inserted_act_dur,
                  act_address, act_event, chatting_with, chat, chatting_with_buffer, chatting_end_time,
                  act_pronunciatio, act_obj_description, act_obj_pronunciatio, act_obj_event)


def plan(persona, maze, personas, new_day, retrieved):
    """
  Main cognitive function of the chain. It takes the retrieved memory and 
  perception, as well as the maze and the first day state to conduct both 
  the long term and short term planning for the persona. 

  INPUT: 
    maze: Current <Maze> instance of the world. 
    personas: A dictionary that contains all persona names as keys, and the 
              Persona instance as values. 
    new_day: This can take one of the three values. 
      1) <Boolean> False -- It is not a "new day" cycle (if it is, we would
         need to call the long term planning sequence for the persona). 
      2) <String> "First day" -- It is literally the start of a simulation,
         so not only is it a new day, but also it is the first day. 
      2) <String> "New day" -- It is a new day. 
    retrieved: dictionary of dictionary. The first layer specifies an event,
               while the latter layer specifies the "curr_event", "events", 
               and "thoughts" that are relevant.
  OUTPUT 
    The target action address of the persona (persona.scratch.act_address).
  """
    # PART 1: Generate the hourly schedule.
    if new_day:
        _long_term_planning(persona, new_day)

    # PART 2: If the current action has expired, we want to create a new plan.
    if persona.scratch.act_check_finished():
        _determine_action(persona, maze)

    # PART 3: If you perceived an event that needs to be responded to (saw
    # another persona), and retrieved relevant information.
    # Step 1: Retrieved may have multiple events represented in it. The first
    #         job here is to determine which of the events we want to focus
    #         on for the persona.
    #         <focused_event> takes the form of a dictionary like this:
    #         dictionary {["curr_event"] = <ConceptNode>,
    #                     ["events"] = [<ConceptNode>, ...],
    #                     ["thoughts"] = [<ConceptNode>, ...]}
    focused_event = False
    if retrieved.keys():
        focused_event = _choose_retrieved(persona, retrieved)

    # Step 2: Once we choose an event, we need to determine whether the
    #         persona will take any actions for the perceived event. There are
    #         three possible modes of reaction returned by _should_react.
    #         a) "chat with {target_persona.name}"
    #         b) "react"
    #         c) False
    if focused_event:
        reaction_mode = _should_react(persona, focused_event, personas)
        if reaction_mode:
            # If we do want to chat, then we generate conversation
            if reaction_mode[:9] == "chat with":
                _chat_react(maze, persona, focused_event, reaction_mode, personas)
            elif reaction_mode[:4] == "wait":
                _wait_react(persona, reaction_mode)
            # elif reaction_mode == "do other things":
            #   _chat_react(persona, focused_event, reaction_mode, personas)

    # Step 3: Chat-related state clean up.
    # If the persona is not chatting with anyone, we clean up any of the
    # chat-related states here.
    if persona.scratch.act_event[1] != "chat with":
        persona.scratch.chatting_with = None
        persona.scratch.chat = None
        persona.scratch.chatting_end_time = None
    # We want to make sure that the persona does not keep conversing with each
    # other in an infinite loop. So, chatting_with_buffer maintains a form of
    # buffer that makes the persona wait from talking to the same target
    # immediately after chatting once. We keep track of the buffer value here.
    curr_persona_chat_buffer = persona.scratch.chatting_with_buffer
    for persona_name, buffer_count in curr_persona_chat_buffer.items():
        if persona_name != persona.scratch.chatting_with:
            persona.scratch.chatting_with_buffer[persona_name] -= 1

    return persona.scratch.act_address
