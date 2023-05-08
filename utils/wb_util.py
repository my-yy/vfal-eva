# used for late-initialize wandb in case of crash before a full evaluation (which add an idle log on the monitoring panel)
import collections
import wandb
import os
from utils import pickle_util

cache = collections.defaultdict(list)

is_inited = False


def save(the_path):
    if is_inited:
        wandb.save(the_path)
    else:
        cache["save"].append(the_path)


def update_config(obj):
    if is_inited:
        wandb.config.update(obj)
    else:
        cache["config"].append(obj)


def log(obj):
    if is_inited:
        wandb.log(obj)
    else:
        cache["log"].append(obj)


def init(args):
    init_core(args.project, args.name, args.dryrun)


def init_core(project, name, dryrun):
    global is_inited
    if is_inited:
        return
    is_inited = True

    if dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
        wandb.log = do_nothing
        wandb.save = do_nothing
        wandb.watch = do_nothing
        wandb.config = {}
        print("wb dryrun mode")
        return

    init_based_on_config_file(project, name)


def init_based_on_config_file(project, name, config_path=".wb_config.json"):
    assert os.path.exists(config_path)
    json_dict = pickle_util.read_json(config_path)

    # use self-hosted wb server
    key = "WANDB_BASE_URL"
    if key in json_dict:
        os.environ[key] = json_dict[key]

    # login
    wandb.login(key=json_dict["WB_KEY"])
    wandb.init(project=project, name=name)
    print("wandb inited")

    # supplement config and logs
    for obj in cache["config"]:
        wandb.config.update(obj)

    for log in cache["log"]:
        wandb.log(log)

    for the_path in cache["save"]:
        wandb.save(the_path)


def do_nothing(v):
    pass
