import ipdb

from utils import pickle_util, vec_util, path_util
from utils.eva_emb_full import EmbEva


def load_face_emb_dict():
    face_emb_dict = pickle_util.read_pickle(path_util.look_up("./dataset/face_input.pkl"))
    vec_util.dict2unit_dict_inplace(face_emb_dict)
    # 2.trans key
    face_emb_dict2 = {}
    for key, v in face_emb_dict.items():
        # 'A.J._Buckley/1.6/1zcIwhmdeo4/0000375.jpg' ==> 'A.J._Buckley/1zcIwhmdeo4/0000375.jpg'
        face_emb_dict2[key.replace("/1.6/", "/")] = v
    return face_emb_dict2


def load_voice_emb_dict():
    voice_emb_dict = pickle_util.read_pickle(path_util.look_up("./dataset/voice_input.pkl"))
    vec_util.dict2unit_dict_inplace(voice_emb_dict)
    name2voice_id = pickle_util.read_pickle("./dataset/info/name2voice_id.pkl")
    voiceid2name = {}
    for k, v in name2voice_id.items():
        voiceid2name[v] = k

    voice_emb_dict2 = {}
    for k, v in voice_emb_dict.items():
        # id11194/bdFSAep9GQk/00005.wav ==> Ty_Pennington/bdFSAep9GQk/00005.wav
        the_id = k.split("/")[0]
        name = voiceid2name[the_id]
        k2 = k.replace(the_id, name)
        voice_emb_dict2[k2] = v
    return voice_emb_dict2


face_emb_dict = load_face_emb_dict()
voice_emb_dict = load_voice_emb_dict()

# project_name = "VFALBenchmark"
# total_epoch = 100
# batch_size = 256
# early_stop = 5
# batch_per_epoch = 500
# eval_step = 150
# save_folder = ""


# 2.eval
# emb_eva = EmbEva(voice_emb_dict, face_emb_dict)
