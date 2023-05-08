from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
import torch
from loaders import v2_unsup_loader
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, _ = data
    v_emb, f_emb = model(voice_data, face_data)
    loss = pair_selection_util.contrastive_loss(f_emb, v_emb, args.margin, tau_value)
    loss.backward()
    optimizer.step()
    info = {
        "train/tau_value": tau_value,
    }
    return loss.item(), info


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

            global tau_value
            if step > 0 and step % 500 == 0 and tau_value < 0.8:
                tau_value = tau_value + 0.1
                print("Update tau:", tau_value)


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=5)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
        "margin": 0.6,
        "tau": 0.3,
    })
    parser.use_wb("VFALBenchmark", "Pins")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = v2_unsup_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    tau_value = args.tau

    # model
    model = Encoder().cuda()
    model_params = model.parameters()

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
