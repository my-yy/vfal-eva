from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.cae_model import CAE
import torch
from loaders import v2_unsup_loader
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, _ = data
    loss_emb, loss_dec = model(voice_data, face_data)
    loss = loss_emb + loss_dec
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
    })
    parser.use_wb("VFALBenchmark", "CAE")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = v2_unsup_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    # model
    model = CAE().cuda()
    model_params = model.parameters()

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
