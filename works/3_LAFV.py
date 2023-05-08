from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
import torch
from loaders import v3_triplet_loader
from utils.losses import triplet_lafv
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva


def check_nan(model):
    # 打印出含有nan的权重层
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"{name} contains NaN values.")


def print_max_val(model):
    # 打印出出权重层中最大数值与最小数值
    max_vas = []
    min_vals = []
    for name, param in model.named_parameters():
        max_vas.append(param.max().item())
        min_vals.append(param.min().item())
    print(max(max_vas), min(min_vals))


def print_max_grad(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            grad = param.grad
            if grad is not None:
                print(name, torch.max(grad).item())


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data_pos, _, face_data_neg = data
    v_emb, f_emb_pos = model(voice_data, face_data_pos)
    f_emb_neg = model.face_encoder(face_data_neg)
    loss = triplet_lafv.triplet_loss(v_emb, f_emb_pos, f_emb_neg)
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
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=5, worker=4)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
    })
    parser.use_wb("VFALBenchmark", "LAFV")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = v3_triplet_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    # model
    model = Encoder().cuda()
    model_params = model.parameters()

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
