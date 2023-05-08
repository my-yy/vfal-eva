from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
import torch
from loaders import v1_sup_id_loader
from utils.losses import unsup_nce
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, id_label, gender_label = data
    v_emb, f_emb = model(voice_data, face_data)
    loss = npair_loss(f_emb, v_emb)
    loss.backward()
    optimizer.step()
    return loss.item(), {}


def train():
    step = 0
    model.train()

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})
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

            if step > 0 and step % args.eval_step == 0:
                if eval_cut.eval_short_cut():
                    return


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=5)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
        "k": -1  # 无意义
    })
    parser.use_wb("VFALBenchmark", "FV-CME")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = v1_sup_id_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    # model
    model = Encoder().cuda()
    model_params = model.parameters()

    npair_loss = unsup_nce.InfoNCE(temperature=1.0)

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
