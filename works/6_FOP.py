from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
from models import fop_model
import torch
from loaders import v1_sup_id_loader
from utils.losses.softmax_loss import SoftmaxLoss
from utils.losses import fop_loss
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, id_label, _ = data
    v_emb, f_emb, fusion_emb = model(voice_data, face_data)

    if args.use_fusion_block:
        loss_id = fun_id_classifier(fusion_emb, id_label)
        loss_fop = fun_fop_loss(fusion_emb, id_label)
    else:
        cat_emb = torch.cat([v_emb, f_emb], dim=0)
        cat_id = torch.cat([id_label, id_label], dim=0).squeeze()
        loss_id = fun_id_classifier(cat_emb, cat_id)
        loss_fop = fun_fop_loss(cat_emb, cat_id)

    loss = loss_id + loss_fop
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
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=6)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 200,
        "use_fusion_block": False,
    })
    parser.use_wb("VFALBenchmark", "FOP")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    wb_util.save(__file__)
    wb_util.save(fop_model.__file__)
    train_iter = v1_sup_id_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    # model
    model = fop_model.FopModel().cuda()

    # loss
    num_class = len(train_iter.dataset.train_names)
    fun_id_classifier = SoftmaxLoss(128, num_class=num_class).cuda()
    fun_fop_loss = fop_loss.OrthogonalProjectionLoss()

    # optimizer
    model_params = list(model.parameters()) + list(fun_id_classifier.parameters())
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
