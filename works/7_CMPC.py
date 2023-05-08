from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
import torch
from loaders import v2_unsup_loader
from utils.losses import cmpc_loss
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva
from torch.nn import functional as F
# from utils import keops_kmeans


class Memory():

    def __init__(self, vec_number, embedding_dim, momentum=0.5):
        super(Memory, self).__init__()
        self.momentum = momentum
        memo = torch.randn(vec_number, embedding_dim).cuda()
        memo = F.normalize(memo, p=2, dim=1)
        self.memo = memo

    def update(self, emb, label):
        with torch.no_grad():
            emb = F.normalize(emb, p=2, dim=1)
            old_cache = self.memo.index_select(0, label)
            old_cache.mul_(self.momentum)
            old_cache.add_(torch.mul(emb, 1 - self.momentum))
            new_cache = F.normalize(old_cache, p=2, dim=1)
            self.memo.index_copy_(0, label, new_cache)


def do_step(data, v_cluster_result, f_cluster_result):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, movie_id = data
    v_emb, f_emb = model(voice_data, face_data)
    v_memory.update(v_emb, movie_id)
    f_memory.update(v_emb, movie_id)

    loss = loss_fun(v_emb, f_emb, v_cluster_result, f_cluster_result, movie_id)
    loss.backward()
    optimizer.step()
    return loss.item(), {}


def train():
    step = 0
    model.train()
    f_cluster_result = None
    v_cluster_result = None

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})

        for data in train_iter:
            loss, info = do_step(data, v_cluster_result, f_cluster_result)
            step += 1
            if step % 50 == 0:
                obj = {
                    "train/step": step,
                    "train/loss": loss,
                }
                obj = {**obj, **info}
                print(obj)
                wb_util.log(obj)

            if step > 0 and step % args.eval_step == 0:
                if eval_cut.eval_short_cut():
                    return

                # do cluster
                # model.eval()
                # num_cluster = [50, 1000, 1500]
                # f_cluster_result = keops_kmeans.run_kmeans(f_memory.memo, num_cluster, Niter=20, temperature=0.2, verbose=True)
                # v_cluster_result = keops_kmeans.run_kmeans(v_memory.memo, num_cluster, Niter=20, temperature=0.2, verbose=True)
                # model.train()


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=6)
    parser.custom({
        "warmup_iter": 50,
        "batch_per_epoch": 500,
        "eval_step": 200,
        "margin": 0.6,
        "use_fusion_block": False,
        "temperature": 0.03,
    })
    parser.use_wb("9.17CMPC", "run1")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    wb_util.save(__file__)
    train_iter = v2_unsup_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    # model
    len_train_movies = len(train_iter.dataset.train_movies)
    v_memory = Memory(len_train_movies, 128)
    f_memory = Memory(len_train_movies, 128)
    model = Encoder().cuda()

    # loss
    loss_fun = cmpc_loss.IR_CMPC(args.temperature, delta=-1, ka=0.1, R=3).cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
