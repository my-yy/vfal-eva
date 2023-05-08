import os


def look_up(path):
    if os.path.exists(path):
        return path

    upper = "." + path
    if os.path.exists(upper):
        print("switch", path, "==>", upper)
        return upper

    return path


import pathlib
def mk_parent_dir_if_necessary(img_save_path):
    folder = pathlib.Path(img_save_path).parent
    if not os.path.exists(folder):
        os.makedirs(folder)
