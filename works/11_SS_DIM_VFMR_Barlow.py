from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full
import os
from models.my_model import Encoder
import torch
from loaders import v1_sup_id_loader, v2_unsup_loader
from utils.losses.softmax_loss import SoftmaxLoss
from utils.losses import triplet_hq1, center_loss_learnableW_L2dist, center_loss_eccv16
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva
from pytorch_metric_learning import losses
from utils.losses import barlow_loss


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, id_label, gender_label = data
    v_emb, f_emb = model(voice_data, face_data)

    cat_emb = torch.cat([v_emb, f_emb], dim=0)
    cat_id = torch.cat([id_label, id_label], dim=0).squeeze()

    if args.name.startswith("SSNet"):
        loss_center = fun_center_loss(cat_emb, cat_id)
        loss_id = fun_id_classifier(cat_emb, cat_id)
        loss = loss_center + loss_id
    elif args.name.startswith("DIMNet"):
        cat_gender_label = torch.cat([gender_label, gender_label], dim=0).squeeze()
        loss_id = fun_id_classifier(cat_emb, cat_id)
        loss_gender = fun_id_classifier(cat_emb, cat_gender_label)
        loss = loss_gender + loss_id
    elif args.name.startswith("LAFV") or args.name.startswith("VFMR"):
        loss = fun_loss_metric(cat_emb, cat_id)
    elif args.name.startswith("SL-Barlow"):
        loss = fun_barlow(v_emb, f_emb)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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
        "margin": 0.6,
        "clip": 1.0
    })
    parser.use_wb("VFALBenchmark", "HQ1_")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = v1_sup_id_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    # model
    model = Encoder().cuda()
    model_params = model.parameters()

    # loss
    num_class = len(train_iter.dataset.train_names)
    if args.name.startswith("SSNet"):
        # 只能达到75%，暂时放弃
        # http://wbz.huacishu.com/cgy/VFALBenchmark/runs/0r6ippc4?workspace=user-chenguangyu
        fun_center_loss = center_loss_eccv16.CenterLoss(num_classes=num_class, feat_dim=128).cuda()
        fun_id_classifier = SoftmaxLoss(128, num_class=num_class).cuda()
        model_params = list(model.parameters()) + list(fun_id_classifier.parameters()) + list(fun_center_loss.parameters())
    elif args.name.startswith("DIMNet"):
        fun_id_classifier = SoftmaxLoss(128, num_class=num_class).cuda()
        fun_gender_classifier = SoftmaxLoss(128, num_class=2).cuda()
        model_params = list(model.parameters()) + list(fun_gender_classifier.parameters()) + list(fun_id_classifier.parameters())
    elif args.name.startswith("VFMR"):
        fun_loss_metric = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)
        model_params = list(model.parameters()) + list(fun_loss_metric.parameters())
    elif args.name.startswith("SL-Barlow"):
        fun_barlow = barlow_loss.BarlowTwinsLoss()
    else:
        raise Exception("Not Support Name:", args.name)

    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
