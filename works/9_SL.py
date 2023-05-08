from utils import my_parser, seed_util, wb_util, deepcluster_util, pickle_util
from utils.eval_shortcut import Cut
from models import my_model
import torch
from loaders import v5_voxceleb_cluster_ordered_loader
from loaders import v6_voxceleb_loader_for_deepcluster
from pytorch_metric_learning import losses
from utils import model_util
from utils.eva_emb_full import EmbEva
import os
from utils.config import face_emb_dict, voice_emb_dict
import ipdb
import tqdm


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, label = data
    v_emb, f_emb = model(voice_data, face_data)
    emb = torch.cat([v_emb, f_emb], dim=0)
    label2 = torch.cat([label, label], dim=0).squeeze()

    if args.ratio_mse > 0:
        loss_mse = fun_loss_mse(v_emb, f_emb) * args.ratio_mse
    else:
        loss_mse = 0

    loss = fun_loss_metric(emb, label2) + loss_mse
    loss.backward()
    optimizer.step()
    info = {
    }
    return loss.item(), info


def get_ratio(loss, total_loss):
    if type(loss) == torch.Tensor:
        loss = loss.item()
    return loss / total_loss.item()


def train():
    step = 0
    model.train()

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})
        # do cluster
        all_keys, all_emb, all_emb_v, all_emb_f = v5_voxceleb_cluster_ordered_loader.extract_embeddings(face_emb_dict, voice_emb_dict, model)
        movie2label, _ = deepcluster_util.do_cluster_v2(all_keys, all_emb, all_emb_v, all_emb_f, args.ncentroids, input_emb_type=args.cluster_type)
        # create dataset
        train_iter = v6_voxceleb_loader_for_deepcluster.get_iter(args.batch_size,
                                                              args.batch_per_epoch * args.batch_size,
                                                              face_emb_dict,
                                                              voice_emb_dict,
                                                              movie2label)

        for data in train_iter:
            loss, info = do_step(epo, step, data)
            step += 1
            if step % 50 == 0:
                obj = {
                    "train/step": step,
                    "train/loss": loss,
                }
                obj = {**obj, **info}
                print(obj)
                wb_util.log(obj)

            if step % args.eval_step == 0:
                if eval_cut.eval_short_cut():
                    return


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=10)
    parser.custom({
        "ncentroids": 1000,
        "batch_per_epoch": 500,
        "eval_step": 200,

        "ratio_mse": 0.0,

        "mts_alpha": 2.0,
        "mts_beta": 50.0,
        "mts_base": 1.0,

        "load_model": "",

        "cluster_type": "all",
    })
    parser.use_wb("sl_project", "SL")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    assert args.cluster_type in ["v", "f", "all"]


    # 1.model:
    model = my_model.Encoder().cuda()
    if args.load_model is not None and os.path.exists(args.load_model):
        model_util.load_model(args.load_model, model, strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3.loss
    fun_loss_metric = losses.MultiSimilarityLoss(alpha=args.mts_alpha, beta=args.mts_beta, base=args.mts_base)

    fun_loss_mse = torch.nn.MSELoss()
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)

    train()


