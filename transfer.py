from torch import load
from cfg import get_cfg
from datasets import get_ds
from methods import get_method

base_model_path = r"./base_model_bt.pt"


if __name__ == "__main__":
    cfg = get_cfg()
    
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)
    model = get_method(cfg.method)(cfg)
    # eval mode
    model.cuda().eval()
    model.load_state_dict(load(base_model_path))
    
    acc_knn, acc = model.get_acc(ds.clf, ds.test)
    print({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn})
    
