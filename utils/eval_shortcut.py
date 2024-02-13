import os
from utils import wb_util, model_util, pickle_util, my_git
from utils import model_selector


class Cut():

    def __init__(self, emb_eva, model, args):
        self.modelSelector = model_selector.ModelSelector()
        self.emb_eva = emb_eva
        self.model = model
        self.args = args
        self.best_obj = None

    def eval_short_cut(self, test_threshold=80):
        emb_eva = self.emb_eva
        model = self.model
        modelSelector = self.modelSelector
        args = self.args

        # 1.do validation
        valid_obj = emb_eva.do_valid(model)
        modelSelector.log(valid_obj)
        indicator = "valid/auc"

        if valid_obj[indicator] < test_threshold:
            obj = valid_obj
            print("model too weak, skip test")
        elif modelSelector.is_best_model(indicator):
            # 2. do test
            test_obj = emb_eva.do_full_test(model, gender_constraint=True)
            obj = {**valid_obj, **test_obj}
            model_util.delete_last_saved_model()
            model_save_name = "auc[%.2f,%.2f]_ms[%.2f,%.2f]_map[%.2f,%.2f].pkl" % (
                obj["valid/auc"],
                obj["test/auc"],
                obj["test/ms_v2f"],
                obj["test/ms_f2v"],
                obj["test/map_v2f"],
                obj["test/map_f2v"],
            )
            model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_save_name)
            model_util.save_model(0, model, None, model_save_path)
            pickle_util.save_json(model_save_path + ".json", test_obj)
            self.best_obj = obj
        else:
            obj = valid_obj
            print("not best model")

        # 2.log
        wb_util.log(obj)
        print(obj)
        wb_util.init(args)
        # my_git.commit_v2(args)

        if modelSelector.should_stop(indicator, args.early_stop):
            print("early_stop")
            if len(model_util.history_array) > 0:
                print(model_util.history_array[-1])
                # 上传best信息
                best_obj = {}
                for k, v in self.best_obj.items():
                    best_obj["best_" + k] = v
                wb_util.log(best_obj)
            return True
        return False
