import ipdb

from utils import my_parser, seed_util, wb_util, pair_selection_util, eva_emb_full, pickle_util, path_util
import os
from models.my_model import Encoder
import torch
from loaders import v1_sup_id_loader
from utils.losses.wen_reweight import *
from utils.losses import wen_explicit_loss
from utils.losses.softmax_loss import SoftmaxLoss
from utils.eval_shortcut import Cut
from utils.eva_emb_full import EmbEva


class ModelWrapper:

    def __init__(self, train_iter, warmup_step=500):
        self.warmup_step = warmup_step

        model = Encoder().cuda()

        # loss
        num_class = len(train_iter.dataset.train_names)
        self.fun_id_classifier = SoftmaxLoss(128, num_class=num_class, reduction="none").cuda()

        # optimizer
        model_params = list(model.parameters()) + list(self.fun_id_classifier.parameters())
        # self.optimizer = torch.optim.SGD(model_params, lr=1e-2)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1, last_epoch=-1)
        self.optimizer = torch.optim.Adam(model_params, lr=args.lr)
        self.scheduler = None

        self.num_class = num_class
        self.model = model

        # forever iter
        def cycle(dataloader):
            while True:
                for data in dataloader:
                    yield data

        self.train_iter = iter(cycle(train_iter))

    def train_step(self, id2weights=None):
        self.optimizer.zero_grad()
        # data
        data = next(self.train_iter)
        data = [i.cuda() for i in data]
        voice_data, face_data, id_label, _ = data
        # to emb
        v_emb, f_emb = self.model(voice_data, face_data)

        # loss
        loss_metric = 0.5 * wen_explicit_loss.cross_logit(v_emb, f_emb) + 0.5 * wen_explicit_loss.cross_logit(f_emb, v_emb)
        loss_ID_cls = self.fun_id_classifier(v_emb, id_label) + self.fun_id_classifier(f_emb, id_label)

        # ReWeight
        if id2weights is not None and len(id2weights) > 0:
            weights = [id2weights[key] for key in id_label.squeeze().detach().cpu().numpy()]
            weights = torch.FloatTensor(weights).cuda()
            loss = (weights * loss_ID_cls).mean() + (weights * loss_metric).mean()
        else:
            loss = loss_ID_cls.mean() + loss_metric.mean()

        loss.backward()
        self.optimizer.step()

        info = {
            "loss_item": loss.item(),
            "label_list": id_label.detach().cpu().numpy(),
            "loss_list": loss_ID_cls.detach().cpu().numpy(),  # use not-weighted ID-Cls loss, Equation(7) of the original paper.
        }
        return info

    def generate_weights(self):
        identitiy_count = self.num_class
        total_step = 0

        # =====================================================stage1:
        print("Stage1: Warmup...")
        for i in range(self.warmup_step):
            info = self.train_step()
            if i % 25 == 0:
                obj = {
                    "train_pre/loss": info["loss_item"],
                    "train_pre/step": i
                }
                wb_util.log(obj)
                print(obj)
            total_step += 1

        # =====================================================stage2:
        print("Stage2: Calculate Weights....")
        t = 0
        id2weights = {}
        hardness = {}
        while not_zero_count(id2weights) < 0.9 * identitiy_count:
            info = self.train_step(id2weights)
            hardness = update_hardness(hardness, info["label_list"], info["loss_list"])

            total_step += 1
            t += 1
            if t % 100 == 0:
                ratio = not_zero_count(id2weights) / identitiy_count

                if self.scheduler is not None:
                    cur_lr = self.scheduler.get_last_lr()[0]
                else:
                    cur_lr = -1

                obj = {
                    "train_pre/step": total_step,
                    "train_pre/lr": cur_lr,
                    "train_pre/loss": info["loss_item"],
                    "train_pre/non_zero_weight_ratio": ratio,
                }
                wb_util.log(obj)
                print(obj)
                id2weights = update_weight(id2weights, hardness)

            if total_step in [2000, 3000] and self.scheduler is not None:
                self.scheduler.step()

        print("Stage2 End, total step:%d" % (total_step))
        return id2weights


def train(id2weights):
    step = 0
    model_wrapper.model.train()
    for i in range(10 * 10000):
        info = model_wrapper.train_step(id2weights)
        info = {"train/loss": info["loss_item"]}

        step += 1
        if step % 50 == 0:
            obj = {
                "train/step": step,
                "train/loss": info["train/loss"],
            }
            print(obj)
            wb_util.log(obj)

        if step > 0 and step % args.eval_step == 0:
            if eval_cut.eval_short_cut():
                return
            model_wrapper.model.train()


def load_wens_official():
    name2weights = {}
    dataset = train_iter.dataset
    lines = [l.strip() for l in open("./dataset/info/works/wen_weights.txt").readlines() if l.strip()]
    for line in lines:
        k, v = line.split(" ")
        name2weights[k] = float(v)

    id2weights = {}
    for name, id in dataset.name2id.items():
        v = name2weights.get(name, 1.0)
        id2weights[id] = v

    for k, v in id2weights.items():
        print(k, v)
    return id2weights


if __name__ == "__main__":
    # python 5_Wen.py --calc_weight=False --load_weight_path="./outputs/VFALBenchmark/use_weight/id2weights.json" --name=use_weight --dryrun=False
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=5)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
        "mode": "load_official",  # load_official、clac_weight、load_file
        "load_weight_path": ""
    })
    parser.use_wb("9.16_Wen", "run1")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    wb_util.init(args)
    wb_util.save(__file__)

    # train
    train_iter = v1_sup_id_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)

    if args.mode == "clac_weight":
        # calc identity weights
        train_iter2 = v1_sup_id_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)
        model_wrapper = ModelWrapper(train_iter2, warmup_step=500)
        id2weights = model_wrapper.generate_weights()
        id2weights_save_path = os.path.join(args.model_save_folder, args.project, args.name, "id2weights.json")
        path_util.mk_parent_dir_if_necessary(id2weights_save_path)
        pickle_util.save_json(id2weights_save_path, id2weights)
        print("id2weights:", id2weights)
    elif args.mode == "load_file":
        print("load weights")
        tmp = pickle_util.read_json(args.load_weight_path)
        id2weights = {}
        for k, v in tmp.items():
            id2weights[int(k)] = v
    elif args.mode == "load_official":
        id2weights = load_wens_official()
    else:
        print("不使用weight")
        id2weights = {}

    model_wrapper = ModelWrapper(train_iter)
    emb_eva = EmbEva()
    eval_cut = Cut(emb_eva, model_wrapper.model, args)
    train(id2weights)
