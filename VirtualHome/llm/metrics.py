import os.path
import json
import inspect


def get_caller_function_names():
    stack = inspect.stack()
    caller_names = [frame.function for frame in stack][2:]
    return '.'.join(caller_names)


class Metrics:
    def __init__(self):
        self.save_fold = None
        # call record
        # function name -> count, token, time
        self.function_name_count = {}
        self.function_name_token = {}
        self.function_name_time = {}
        # model -> count, token
        self.model_count = {}
        self.model_token = {}
        # fail record
        # function name -> fail count, reason
        self.function_name_fail_count = {}
        self.function_name_fail_reason = {}
        # detailed info
        self.detail_info = []

    def set_fold(self, path):
        fold_name = 'metrics'
        save_path = os.path.join(path, fold_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.save_fold = save_path

    def save(self):
        assert self.save_fold is not None, "error: you should set the metrics save path."
        save_pair = [
            [self.function_name_count, 'function_name_count.json'],
            [self.function_name_token, 'function_name_token.json'],
            [self.function_name_time, 'function_name_time.json'],
            [self.model_count, 'model_count.json'],
            [self.model_token, 'model_token.json'],
            [self.function_name_fail_count, 'function_name_fail_count.json'],
            [self.function_name_fail_reason, 'function_name_fail_reason.json'],
            [self.detail_info, 'detail_info.json'],
        ]
        for dict_data, file_name in save_pair:
            json_str = json.dumps(dict_data, indent=4)
            with open(os.path.join(self.save_fold, file_name), 'w') as json_file:
                json_file.write(json_str)

    def call_record(self, function_name, model, token, time):
        if function_name not in self.function_name_count.keys():
            self.function_name_count[function_name] = 0
            self.function_name_time[function_name] = []
            self.function_name_token[function_name] = []
        if model not in self.model_count.keys():
            self.model_count[model] = 0
            self.model_token[model] = []

        self.function_name_count[function_name] += 1
        self.function_name_token[function_name].append(token)
        self.function_name_time[function_name].append(time)
        self.model_count[model] += 1
        self.model_token[model].append(token)

        self.detail_info.append(
            {
                "function_name": function_name,
                "model": model,
                "token": token,
                "time": time,
            }
        )

    def fail_record(self, reason):
        function_name = get_caller_function_names()
        if function_name not in self.function_name_fail_count.keys():
            self.function_name_fail_count[function_name] = 0
            self.function_name_fail_reason[function_name] = []
        self.function_name_fail_count[function_name] += 1
        self.function_name_fail_reason[function_name].append(str(reason))
        print(f"FAIL_RECORD: function:{function_name} reason:{reason}")


metrics = Metrics()
