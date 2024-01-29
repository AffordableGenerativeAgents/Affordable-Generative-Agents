# AGA in Generative Agents

<p align="center" width="100%">
<img src="../doc/pic/aga_in_ga.gif" width="50%" height="50%">
</p>

This work is based on [Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents). Generative Agents provides a platform that simulates a virtual town with both front-end and back-end capabilities. For convenience and to reduce experiment time, we offer a version that operates purely on the back-end. For more detailed information about the platform, please refer to [Generative Agents](https://github.com/joonspk-research/generative_agents).

## Preparation
To set up your environment, you will need to generate a `utils.py` file that contains your LLM API key and download the necessary packages.

### Step 1. Generate Utils File
In the `reverie/backend_server` folder (where `reverie.py` is located), create a new file titled `utils.py` and copy and paste the content below into the file, we offer Azure API, OpenAI API and llama version (you should specify the `key_type`):
```python
import os

key_type = 'azure'
assert key_type in ['openai', 'azure', 'llama'], "ERROR: wrong key type, the key type should select from ['openai', " \
                                                 "'azure', 'llama']. "
# openai
openai_api_key = "<Your OpenAI API>"
key_owner = "<Name>"

# azure
if key_type == 'azure':
    openai_api_key = "<Your Azure API Key>"
    openai_api_base = "<Your Azure API Base>"  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai_api_type = 'azure'
    openai_api_version = '2023-05-15'  # this may change in the future
    # for completion
    openai_completion_api_key = "<Your Azure API Key>"
    openai_completion_api_base = "<Your Azure API Base>"

# llama
if key_type == 'llama':
    openai_api_key = "none"
    openai_api_base = "<llama URL>"  # The address of the llama model you deployed.
    openai_api_type = 'openai'
    openai_api_version = ''

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
# sim fold
sim_fold = None
def set_fold(path):
    global sim_fold
    sim_fold = path
# Pool
use_embedding_pool = True
embedding_pool_path = os.path.join(fs_storage, "public", "embedding_pool.json")
use_policy_pool = True
policy_pool_path = os.path.join(fs_storage, "public", "policy_pool")
use_sub_task_pool = True
sub_task_pool_path = os.path.join(fs_storage, "public", "sub_task_pool")
# Record
record_tree_flag = True
# switch
use_policy = True
use_relationship = True
```

### Step 2. Install requirements.txt
Install everything listed in the `requirements.txt` file.

## Running a Simulation
We offer the back-end only version in `reverie_offline.py`, you should run in the following format:
```bash
python reverie_offline.py -o <the forked simulation> -t <the new simulation> -s <the total run step>

# Here is an example
python reverie_offline.py -o base_the_ville_isabella_maria_klaus -t aga_3_person -s 17280
```

## Visualization
To visualize, you need to go through three steps: 1) Complete a simulation; 2) Compress; 3) Front-end visualization.

### Step 1. Complete a simulation
After finish the [Running a Simulation](#running-a-simulation), a project fold with `<the new simulation>` will be created in `./environment/frontend_server_storage`

### Step 2. Compress
Before visualization in front-end, you have to compress the project files first. 

change the code in `./reverie/compress_sim_storage.py`

```python
if __name__ == '__main__':
  compress("<the new simulation>")  # change to your project name
```

Run the following command:

```bash
python compress_sim_storage.py
```

### Step 3. Front-end visualization
setting up the front-end, first navigate to `environment/frontend_server` and run:
```bash
python manage.py runserver
```

To start the visualization, go to the following address on your browser: `http://localhost:8000/demo/<the new simulation>/<starting-time-step>/<simulation-speed>`. Note that `<the new simulation>` denote the same things as mentioned above. `<simulation-speed>` can be set to control the demo speed, where 1 is the slowest, and 5 is the fastest. For instance, visiting the following link will start a pre-simulated example, beginning at time-step 1, with a medium demo speed:  
[http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/](http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/)

# Experiment

The **Lifestyle policy** and **Social Impression Memory** are enabled by default. For ablation study, you can turn them off by:
```bash
python reverie_offline.py ... \
    --disable_policy \     # Turn off the Lifestyle policy
    --disable_relationship # Turn off the Social Impression Memory
```

All relevant records for the experiments are generated in the `<project fold>/metrics`:
```
─ metrics
├── detail_info.json                # Complete LLM call log
├── function_name_count.json        # LLM function call count statistics.
├── function_name_fail_count.json   # LLM function call failure count statistics.
├── function_name_fail_reason.json  # LLM function call failure count statistics.
├── function_name_time.json         # Statistics on the reasons for LLM function call failures.
├── function_name_token.json        # Statistics on LLM token consumption by different functions.
├── model_count.json                # LLM model call count statistics.
├── model_token.json                # Statistics on LLM token consumption by different models.
└── <personas_name>.json            # All action logs for the corresponding agent with <personas_name>
```

During experiments, generated policies and embedding features are saved in `./environment/fribtebd_server/storage/public`. The number of policies will affect the token consumption of the experiment.

The implementation of **MindWandering** is in the branch of `mindwandering`