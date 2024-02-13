from deepface import DeepFace
import ipdb
import torch
import time
from torch.utils.data import DataLoader
from .loaders.loader4deepface import Dataset
from utils import pickle_util


def handle_one(num_workers, model_name):
    model = DeepFace.build_model(model_name)

    start_time = time.time()
    batch_size = 128

    the_iter = DataLoader(Dataset(root_path, mode="deepface"),
                          num_workers=num_workers, batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True)
    all_data = the_iter.dataset.all_image_files

    total_batch = int(len(all_data) / batch_size) + 1
    people2emb = {}
    counter = 0
    vec_dim = 0
    with torch.no_grad():
        for image_tensor, indexies in the_iter:
            counter += 1
            # 获取emb：
            batch_emb = model.predict(image_tensor.detach().cpu().numpy())
            assert len(batch_emb.shape) == 2

            indexies = indexies.detach().cpu().numpy().tolist()
            for idx, emb in zip(indexies, batch_emb):
                filepath = all_data[idx]
                # /home/my/datasets/2_VFMR/4_vgg1_mtcnn/Zack_Snyder/0013800.jpg
                tmp_arr = filepath.split("/")
                short_path = tmp_arr[-2] + "/" + tmp_arr[-1]
                people2emb[short_path] = emb
                vec_dim = len(emb)

            time_cost_h = (time.time() - start_time) / 3600.0
            progress = (counter + 1) / total_batch
            full_time = time_cost_h / progress
            print(counter, progress, "full:", full_time)

    save_name = "deepface_%s_dim%d.pkl" % (model_name, vec_dim)

    pickle_util.save_pickle(save_name, people2emb)
    print(save_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    args = parser.parse_args()

    root_path = "./jpgs/"
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace"]
    handle_one(8, models[args.index])
