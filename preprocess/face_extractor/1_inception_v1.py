import time
import torch
from torch.utils.data import DataLoader
from .models import incep
from .loaders.img_loader import Dataset
from utils import pickle_util
import glob

the_dict = {}


def handle_emb_batch(all_data, batch_emb, indexies):
    batch_emb = batch_emb.detach().cpu().numpy().squeeze()
    assert len(batch_emb.shape) == 2
    indexies = indexies.detach().cpu().numpy().tolist()
    for idx, emb in zip(indexies, batch_emb):
        filepath = all_data[idx]
        the_dict[filepath] = emb


def fun(num_workers, all_img_data, batch_size):
    start_time = time.time()
    the_iter = DataLoader(Dataset(all_img_data), num_workers=num_workers, batch_size=batch_size, shuffle=False,
                          pin_memory=True)
    all_data = the_iter.dataset.all_image_files

    total_batch = int(len(all_data) / batch_size) + 1
    counter = 0
    with torch.no_grad():
        for image_tensor, indexies in the_iter:
            counter += 1
            emb_vec = model(image_tensor.cuda())
            handle_emb_batch(all_data, emb_vec, indexies)
            time_cost_h = (time.time() - start_time) / 3600.0
            progress = (counter + 1) / total_batch
            full_time = time_cost_h / progress
            print(counter, progress, "full:", full_time)


if __name__ == '__main__':
    # 1.load model
    model = incep.InceptionResnetV1(pretrained="vggface2", classify=True)
    model.cuda()
    model.eval()

    # 2.get all img list
    all_jpgs = glob.glob("/your_path/*.jpg")

    # 3.processing
    fun(8, all_jpgs, batch_size=2048)

    # 4.save
    pickle_util.save_pickle("face_emb.pkl", the_dict)
