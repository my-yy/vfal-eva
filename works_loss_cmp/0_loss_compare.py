from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
import torch
from loaders import v1_sup_id_loader, v2_unsup_loader
from utils.losses import triplet_hq1, center_loss_learnableW_L2dist, center_loss_eccv16
from utils.eval_shortcut import Cut
from utils import eva_emb_full
from utils.losses import fop_loss, my_pml_infonce_v2
from utils.losses import barlow_loss
from utils.losses import unsup_nce
from utils.losses.softmax_loss import SoftmaxLoss
from pytorch_metric_learning import losses


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    if len(data) == 4:
        voice_data, face_data, id_label, _ = data
    else:
        assert len(data) == 3
        voice_data, face_data, _ = data

    v_emb, f_emb = model(voice_data, face_data)

    if args.loss == "barlow":
        loss = loss_fun(v_emb, f_emb)
    elif args.loss == "unsup_nce":
        loss = loss_fun(v_emb, f_emb) + loss_fun(f_emb, v_emb)
    elif args.loss == "hq_triplet":
        loss = triplet_hq1.triplet_loss(v_emb, f_emb, id_label)
    elif args.loss == "sup_nce_v2":
        loss = loss_fun(v_emb, f_emb, id_label, id_label) + loss_fun(f_emb, v_emb, id_label, id_label)
    else:
        cat_emb = torch.cat([v_emb, f_emb], dim=0)
        cat_id = torch.cat([id_label, id_label], dim=0).squeeze()
        loss = loss_fun(cat_emb, cat_id)

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=10)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
        "loss": "sup_nce",
        "contrastive_margin": 0.5,
        "triplet_margin": 1.0,
        "infoNCE_temperature": 0.07,
        "mts_alpha": 2.0,
        "mts_beta": 50.0,
        "mts_base": 1.0,
    })
    parser.use_wb("VFALBenchmark", "LossComp")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    # data
    if args.loss in ["unsup_nce", "barlow"]:
        # use unsupervised loader for unsupvised loss function!
        train_iter = v2_unsup_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)
    else:
        train_iter = v1_sup_id_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)
        num_class = len(train_iter.dataset.train_names)

    # model
    model = Encoder().cuda()

    # loss
    if args.loss == "sup_nce_pml":
        loss_fun = losses.NTXentLoss(temperature=args.infoNCE_temperature)
    elif args.loss == "sup_nce_v2":
        loss_fun = my_pml_infonce_v2.InfoNCE(temperature=args.infoNCE_temperature, reduction="mean")
    elif args.loss == "unsup_nce":
        loss_fun = unsup_nce.InfoNCE(args.infoNCE_temperature)
    elif args.loss == "pml_triplet_loss":
        loss_fun = losses.TripletMarginLoss()
    elif args.loss == "NCALoss":
        loss_fun = losses.NCALoss(softmax_scale=1)
    elif args.loss == "ProxyNCA":
        loss_fun = losses.ProxyNCALoss(num_classes=num_class, embedding_size=128).to(torch.device('cuda'))
    elif args.loss == "MultiSimilarity":
        loss_fun = losses.MultiSimilarityLoss(alpha=args.mts_alpha, beta=args.mts_beta, base=args.mts_base)
    elif args.loss == "LiftedStructure":
        loss_fun = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)
    elif args.loss == "barlow":
        loss_fun = barlow_loss.BarlowTwinsLoss()
    elif args.loss == "softmax":
        loss_fun = SoftmaxLoss(128, num_class=num_class).cuda()
    elif args.loss == "fop":
        loss_fun = fop_loss.OrthogonalProjectionLoss()
    else:
        raise Exception("wrong loss function:" + args.loss)

    if loss_fun is not None:
        model_params = list(model.parameters()) + list(loss_fun.parameters())
    else:
        model_params = model.parameters()

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = eva_emb_full.EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
