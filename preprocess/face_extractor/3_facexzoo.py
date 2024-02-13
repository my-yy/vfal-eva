import os
import argparse
from torch.utils.data import DataLoader
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
from utils import pickle_util
from data_processor.test_dataset2_resize112 import CommonTestDataset
from backbone.backbone_def import BackboneFactory


def load_model(model_pkl_folder_root):
    # 1.模型
    backbone_type = args.backbone_type
    backbone_conf_file = "backbone_conf.yaml"
    backbone_factory = BackboneFactory(backbone_type, backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)

    # pkl所在文件夹：
    model_pkl_folder = os.path.join(model_pkl_folder_root, args.model_pkl)
    assert os.path.exists(model_pkl_folder)

    pt_name_list = [i for i in os.listdir(model_pkl_folder) if i.endswith(".pt")]
    assert len(pt_name_list) == 1
    # 加载参数：
    model_path = os.path.join(model_pkl_folder, pt_name_list[0])
    model = model_loader.load_model(model_path)
    return model


def load_dataloader(cropped_face_folder):
    image_list_file_path = "1fps_pathlist.txt"
    batch_size = args.batch_size
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder,
                                               image_list_file_path, args.size, False),
                             batch_size=batch_size, num_workers=8, shuffle=False)

    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone_type", default="AttentionNet")
    parser.add_argument("--model_pkl", default="2_Attention56")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--size", default=112, type=int)
    parser.add_argument("--save_name", default="AttentionNet.pkl")

    args = parser.parse_args()

    assert len(args.save_name) > 0

    model = load_model("/home/my/projects/124_FaceX-Zoo/models")

    data_loader = load_dataloader("faces/")

    # 抽取特征
    feature_extractor = CommonExtractor('cuda:0')
    image_name2feature = feature_extractor.extract_online(model, data_loader)

    # 保存：
    dim_len = -1
    for k, v in image_name2feature.items():
        dim_len = len(v)
        break

    save_name = args.save_name.replace(".pkl", "dim%d.pkl" % (dim_len))
    save_path = os.path.join("/ssd2/1_Voxceleb2/4_faceX_Zoo_extracted_feature/1fps", save_name)
    pickle_util.save_pickle(save_path, image_name2feature)
    print(save_path)
