import torch

from src.utils import load_yaml, set_seed
from src.path import SRC_PATH
from src.data import LightKGDataLoader
from model.lightkg import LightKG, trainer


if __name__ == "__main__":

    # ----------------------------
    # 1. Load config
    # ----------------------------
    config_fpath = SRC_PATH / "config.yaml"
    cfg = load_yaml(config_fpath)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    path_cfg = cfg["path"]

    # ----------------------------
    # 2. Seed
    # ----------------------------
    set_seed(data_cfg.get("seed", 42))

    # ----------------------------
    # 3. Device
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # 4. DataLoader 생성
    # ----------------------------
    print("[INFO] Loading data...")
    dataloader = LightKGDataLoader(
        fname=data_cfg["fname"],
        test_size=data_cfg.get("test_size", 0.2),
        seed=data_cfg.get("seed", 42),
    )

    print(f"[INFO] #users : {dataloader.user_num}")
    print(f"[INFO] #items : {dataloader.item_num}")
    print(f"[INFO] #stores: {dataloader.store_num}")
    print(f"[INFO] #train interactions: {len(dataloader.train_df)}")
    print(f"[INFO] #test  interactions: {len(dataloader.test_df)}")
    print(f"[INFO] #edges in CKG: {dataloader.edge_index.size(1)}")

    # edge_index / edge_type / edge_norm to device
    edge_index = dataloader.edge_index.to(device)
    edge_type = dataloader.edge_type.to(device)
    edge_norm = dataloader.edge_norm.to(device)

    # ----------------------------
    # 5. LightKG 생성
    # ----------------------------
    model = LightKG(
        user_num=dataloader.user_num,
        item_num=dataloader.item_num,
        store_num=dataloader.store_num,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_norm=edge_norm,
        rel_num=dataloader.rel_num,
        embed_dim=model_cfg["embed_dim"],
        n_layer=model_cfg["n_layer"],
        l2_reg=model_cfg["l2_reg"],
        beta_user=model_cfg.get("beta_user", 1e-3),
        beta_item=model_cfg.get("beta_item", 1e-3),
    ).to(device)

    print("[INFO] Model:")
    print(model)

    # ----------------------------
    # 6. Optimizer 생성
    # ----------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
    )

    # ----------------------------
    # 7. Train 루프 실행
    # ----------------------------
    trainer(
        model=model,
        data_loader=dataloader,
        optimizer=optimizer,
        batch_size=train_cfg["batch_size"],
        epoch_num=train_cfg["epoch_num"],
        num_batches_per_epoch=train_cfg["num_batches_per_epoch"],
        eval_interval=train_cfg["eval_interval"],
        eval_k=eval_cfg["k"],
        patience=train_cfg["patience"],
        best_model_path=path_cfg["best_model_path"],
        device=device,
        num_user_pairs=train_cfg.get("num_user_pairs", 512),
        num_item_pairs=train_cfg.get("num_item_pairs", 512),
    )
