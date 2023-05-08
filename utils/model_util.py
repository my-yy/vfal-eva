import torch
import os
import json
import sys
from utils import pickle_util

history_array = []


def save_model(epoch, model, optimizer, file_save_path):
    dirpath = os.path.abspath(os.path.join(file_save_path, os.pardir))
    if not os.path.exists(dirpath):
        print("mkdir:", dirpath)
        os.makedirs(dirpath)

    opti = None
    if optimizer is not None:
        opti = optimizer.state_dict()

    torch.save(obj={
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': opti,
    }, f=file_save_path)

    history_array.append(file_save_path)


def delete_last_saved_model():
    if len(history_array) == 0:
        return
    last_path = history_array.pop()
    if os.path.exists(last_path):
        os.remove(last_path)
        print("delete model:", last_path)

    if os.path.exists(last_path + ".json"):
        os.remove(last_path + ".json")


def load_model(resume_path, model, optimizer=None, strict=True):
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("checkpoint loaded!")
    return start_epoch


def save_model_v2(model, args, model_save_name):
    model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_save_name)
    save_model(0, model, None, model_save_path)
    print("save:", model_save_path)


def save_project_info(args):
    run_info = {
        "cmd_str": ' '.join(sys.argv[1:]),
        "args": vars(args),
    }

    name = "run_info.json"
    folder = os.path.join(args.model_save_folder, args.project, args.name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    json_file_path = os.path.join(folder, name)
    with open(json_file_path, "w") as f:
        json.dump(run_info, f)

    print("save_project_info:", json_file_path)


def get_pkl_json(folder):
    names = [i for i in os.listdir(folder) if ".pkl.json" in i]
    assert len(names) == 1
    json_path = os.path.join(folder, names[0])
    obj = pickle_util.read_json(json_path)
    return obj


# ======================================= 分析
import numpy as np
import os
import json


# 不同seed，结果聚合为一个
# name_start_with: 通过名字来过滤目标文件夹
# 附加了std
def result_ensemble(project_folder, name_start_with, format="mean_std"):
    tmp_arr = []
    for run_name in os.listdir(project_folder):
        if run_name.startswith(name_start_with):  # 排除不对的
            obj = find_json(os.path.join(project_folder, run_name))
            tmp_arr.append(obj)

    assert len(tmp_arr) > 1
    new_obj = {
        "name": name_start_with,
    }
    for key in tmp_arr[0].keys():
        vs = [o[key] for o in tmp_arr]
        mean = np.mean(vs)
        std = np.std(vs)
        if format == "mean_std":
            txt = "%.1f±%.1f" % (mean, std)
        else:
            txt = "%.1f" % (mean)
        new_obj[key] = txt
    return new_obj


def find_json(the_path):
    files = os.listdir(the_path)
    files = [f for f in files if f.endswith('.json')]
    assert len(files) == 1
    with open(os.path.join(the_path, files[0]), 'r') as f:
        json_data = json.load(f)
        return json_data


def array_objs_merge_to_single_obj(array):
    new_obj = {}
    for key in array[0].keys():
        new_obj[key] = float("%.2f" % (np.mean([o[key] for o in array]) * 100))
    return new_obj
