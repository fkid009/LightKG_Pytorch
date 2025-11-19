import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, List

from sklearn.model_selection import train_test_split

from src.utils import getDF
from src.path import DATA_DIR


class LightKGDataLoader:
    def __init__(
        self,
        fname: str,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.fname = fname
        self.fpath = f"{fname}.jsonl.gz"
        self.inter_fpath = DATA_DIR / self.fpath
        self.meta_fpath = DATA_DIR / f"meta_{self.fpath}"

        self.test_size = test_size
        self.seed = seed

        # raw interaction / meta
        self.raw_inter_df = self._load_raw_data(self.inter_fpath, is_inter=True)
        self.meta_df = self._load_raw_data(self.meta_fpath, is_inter=False)

        # relation mapping
        self.rel2id = {
            "Interact": 0,
            "Store": 1,
        }
        self.rel_num = len(self.rel2id)

        (
            self.user2id,
            self.item2id,
            self.store2id,
            self.user_num,
            self.item_num,
            self.store_num,
            self.train_df,
            self.test_df,
        ) = self._preprocess()

        (
            self.edge_index,
            self.edge_type,
            self.deg,
            self.edge_norm,
        ) = self._build_KG()

        self.train_user_pos = self._get_user_pos(self.train_df)
        self.test_user_pos = self._get_user_pos(self.test_df)

    # ---------------------------------------------------------------
    # loader
    # ---------------------------------------------------------------
    def _load_raw_data(self, fpath, is_inter: bool = True) -> pd.DataFrame:
        df = getDF(fpath)

        if is_inter:
            df = df.rename(
                columns={
                    "user_id": "user",
                    "parent_asin": "item",
                }
            )
            return df[["user", "item"]].dropna()
        else:
            df = df.rename(columns={"parent_asin": "item"})
            return df[["item", "store"]].dropna()

    def _get_user_pos(self, df: pd.DataFrame) -> Dict[int, set]:
        user_pos = defaultdict(set)
        for _, (u, i) in df.iterrows():
            user_pos[int(u)].add(int(i))
        return user_pos

    # ---------------------------------------------------------------
    # preprocess
    # ---------------------------------------------------------------
    def _preprocess(self):
        # interactions
        df = self.raw_inter_df.copy()
        df = df.drop_duplicates(subset=["user", "item"])

        # meta (item-store)
        meta_df = self.meta_df.copy()
        # keep only items that appear in interaction
        inter_items = set(df["item"].unique())
        meta_df = meta_df[meta_df["item"].isin(inter_items)]
        meta_df = meta_df.drop_duplicates(subset=["item", "store"])
        self.meta_df = meta_df

        # id mappings
        user2id = {u: idx for idx, u in enumerate(df["user"].unique())}
        item2id = {i: idx for idx, i in enumerate(df["item"].unique())}
        store2id = {s: idx for idx, s in enumerate(meta_df["store"].unique())}

        df["user_idx"] = df["user"].map(user2id)
        df["item_idx"] = df["item"].map(item2id)

        user_num = len(user2id)
        item_num = len(item2id)
        store_num = len(store2id)

        # # store id mapping을 위해 meta_df를 저장해두기 (item, store, idx 매핑용)
        # self.meta_df = meta_df  # filtered 버전으로 교체

        train_df, test_df = train_test_split(
            df[["user_idx", "item_idx"]],
            test_size=self.test_size,
            random_state=self.seed,
        )

        return user2id, item2id, store2id, user_num, item_num, store_num, train_df, test_df

    # ---------------------------------------------------------------
    # KG build
    # ---------------------------------------------------------------
    def _build_KG(self):
        user_offset = 0
        item_offset = user_offset + self.user_num
        store_offset = item_offset + self.item_num
        num_nodes = self.user_num + self.item_num + self.store_num

        head_indices: List[int] = []
        tail_indices: List[int] = []
        rel_indices: List[int] = []

        # item - store edges
        for _, row in self.meta_df.iterrows():
            item_key = row["item"]
            store_name = row["store"]

            if item_key not in self.item2id:
                continue

            i_idx = self.item2id[item_key]
            s_idx = self.store2id[store_name]

            h_idx = item_offset + i_idx
            t_idx = store_offset + s_idx
            r_idx = self.rel2id["Store"]

            head_indices.append(h_idx)
            tail_indices.append(t_idx)
            rel_indices.append(r_idx)

        # user - item edges (Interact)
        for _, (u, i) in self.train_df.iterrows():
            u = int(u)
            i = int(i)

            h_idx = user_offset + u
            t_idx = item_offset + i
            r_idx = self.rel2id["Interact"]

            head_indices.append(h_idx)
            tail_indices.append(t_idx)
            rel_indices.append(r_idx)

        head_indices = np.array(head_indices, dtype=np.int64)
        tail_indices = np.array(tail_indices, dtype=np.int64)
        rel_indices = np.array(rel_indices, dtype=np.int64)

        edge_index = np.vstack([head_indices, tail_indices])  # (2, E)
        edge_type = rel_indices  # (E,)

        edge_index = torch.from_numpy(edge_index)
        edge_type = torch.from_numpy(edge_type)

        # degree (numpy) + torch 변환
        deg_np = np.zeros(num_nodes, dtype=np.float32)
        for h, t in zip(head_indices, tail_indices):
            deg_np[h] += 1.0
            deg_np[t] += 1.0

        deg = torch.from_numpy(deg_np)  # (N,)

        # edge normalization (numpy에서 계산 후 torch로 변환)
        edge_norm_np = 1.0 / np.sqrt(
            deg_np[head_indices] * deg_np[tail_indices] + 1e-8
        ).astype(np.float32)
        edge_norm = torch.from_numpy(edge_norm_np)  # (E,)

        return edge_index, edge_type, deg, edge_norm

    # ---------------------------------------------------------------
    # BPR batch sampler
    # ---------------------------------------------------------------
    def get_bpr_batch(self, batch_size: int):
        users = []
        pos_items = []
        neg_items = []

        all_items = np.arange(self.item_num)
        user_pos = self.train_user_pos

        for _ in range(batch_size):
            # sample user
            u = np.random.choice(list(user_pos.keys()))
            pos_list = list(user_pos[u])

            # sample positive
            i = np.random.choice(pos_list)

            # sample negative
            while True:
                j = np.random.choice(all_items)
                if j not in user_pos[u]:
                    break

            users.append(u)
            pos_items.append(i)
            neg_items.append(j)

        return (
            torch.LongTensor(users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items),
        )
