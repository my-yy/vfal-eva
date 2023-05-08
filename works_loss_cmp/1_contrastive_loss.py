from utils import my_parser, seed_util, wb_util
from models.my_model import Encoder
import torch
from loaders import v7_sup_id_contrastive_loader
from utils.eval_shortcut import Cut
from utils import eva_emb_full


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, is_same_person = data
    v_emb, f_emb = model(voice_data, face_data)
    loss = loss_fun(v_emb, f_emb, is_same_person)
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
                if eval_cut.eval_short_cut(test_threshold=0):
                    return


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder="./outputs/", early_stop=10)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 100,
        "margin": 0.5,
        "loss": ""
    })
    parser.use_wb("VFALBenchmark", "contrastive loss")
    args = parser.parse()
    seed_util.set_seed(args.seed)
    train_iter = v7_sup_id_contrastive_loader.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size)
    num_class = len(train_iter.dataset.train_names)

    # model
    model = Encoder().cuda()

    loss_fun = torch.nn.CosineEmbeddingLoss(margin=args.margin, reduction="mean")

    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    emb_eva = eva_emb_full.EmbEva()
    eval_cut = Cut(emb_eva, model, args)
    train()
