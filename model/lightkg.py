import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightKG(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        store_num: int,
        edge_index: torch.Tensor,  # (2, E)
        edge_type: torch.Tensor,   # (E,)
        edge_norm: torch.Tensor,   # (E,)
        rel_num: int,
        embed_dim: int,
        n_layer: int,
        l2_reg: float,
        beta_user: float = 1e-3,
        beta_item: float = 1e-3,
    ):
        """
        LightKG model (LightGCN + relation-wise scalar weights on CKG + contrastive loss).

        Parameters
        ----------
        user_num : int
            Number of users.
        item_num : int
            Number of items.
        store_num : int
            Number of store (entity) nodes.
        edge_index : LongTensor, shape (2, E)
            Head and tail indices for each edge in the CKG.
        edge_type : LongTensor, shape (E,)
            Relation id for each edge.
        edge_norm : FloatTensor, shape (E,)
            Normalization term 1 / sqrt( deg[h] * deg[t] ).
        rel_num : int
            Number of relation types (e.g., Interact, Store, ...).
        embed_dim : int
            Dimensionality of node embeddings.
        n_layer : int
            Number of propagation layers.
        l2_reg : float
            L2 regularization coefficient for BPR loss.
        beta_user : float
            Weight for user-side contrastive loss L_u.
        beta_item : float
            Weight for item-side contrastive loss L_i.
        """
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.store_num = store_num
        self.node_num = user_num + item_num + store_num

        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.l2_reg = l2_reg
        self.rel_num = rel_num
        self.beta_user = beta_user
        self.beta_item = beta_item

        # Graph structure (buffer: not trainable, moves with .to(device))
        self.register_buffer("edge_index", edge_index.long())   # (2, E)
        self.register_buffer("edge_type", edge_type.long())     # (E,)
        self.register_buffer("edge_norm", edge_norm.float())    # (E,)

        # Node embeddings: users + items + stores
        self.embedding = nn.Embedding(self.node_num, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Relation-wise scalar parameters for both directions
        #   - alpha_head2tail[r]: message from head -> tail for relation r
        #   - alpha_tail2head[r]: message from tail -> head for relation r
        self.alpha_head2tail = nn.Parameter(torch.ones(rel_num))
        self.alpha_tail2head = nn.Parameter(torch.ones(rel_num))

    # ------------------------------------------------------------------
    # Propagation (LightKG GNN)
    # ------------------------------------------------------------------
    def propagate(self):
        """
        LightKG propagation over CKG.

        Returns
        -------
        E_user : (user_num, embed_dim)
        E_item : (item_num, embed_dim)
            Final user/item embeddings after layer combination
            (stores are updated internally but not returned).
        """
        # E^{(0)}: initial node embeddings
        E = self.embedding.weight   # (N, D)
        all_embeddings = [E]

        h = self.edge_index[0]      # (E,)
        t = self.edge_index[1]      # (E,)
        r = self.edge_type          # (E,)
        norm = self.edge_norm       # (E,)

        for _ in range(self.n_layer):
            # Initialize next-layer embeddings as zeros
            E_next = torch.zeros_like(E)

            # ----- tail -> head messages -----
            coeff_t2h = norm * self.alpha_tail2head[r]      # (E,)
            msg_t2h = E[t] * coeff_t2h.unsqueeze(1)        # (E, D)
            E_next.index_add_(0, h, msg_t2h)

            # ----- head -> tail messages -----
            coeff_h2t = norm * self.alpha_head2tail[r]      # (E,)
            msg_h2t = E[h] * coeff_h2t.unsqueeze(1)        # (E, D)
            E_next.index_add_(0, t, msg_h2t)

            # move to next layer
            E = E_next
            all_embeddings.append(E)

        # Layer combination (uniform weighting): E* = sum(E^{(k)}) / (L+1)
        E_all = sum(all_embeddings) / len(all_embeddings)   # (N, D)

        # Split to user/item (stores are ignored for BPR inference)
        E_user = E_all[: self.user_num]
        E_item = E_all[self.user_num : self.user_num + self.item_num]

        return E_user, E_item

    # ------------------------------------------------------------------
    # BPR part (same interface as LightGCN)
    # ------------------------------------------------------------------
    def forward(self, user_idx, item_idx):
        """
        Return user/item embeddings for given indices.
        """
        E_user, E_item = self.propagate()
        return E_user[user_idx], E_item[item_idx]

    def predict(self, user_idx, item_idx):
        """
        Predict preference scores with inner product of final embeddings.
        """
        user_emb, item_emb = self.forward(user_idx, item_idx)
        return (user_emb * item_emb).sum(dim=1)

    def bpr_loss(self, user_idx, pos_idx, neg_idx):
        """
        BPR loss using current LightKG embeddings.

        Parameters
        ----------
        user_idx : LongTensor, shape (B,)
        pos_idx  : LongTensor, shape (B,)
        neg_idx  : LongTensor, shape (B,)

        Returns
        -------
        loss : torch.Tensor
            Scalar BPR loss with L2 regularization.
        """
        E_user, E_item = self.propagate()

        u_e   = E_user[user_idx]   # (B, D)
        pos_e = E_item[pos_idx]    # (B, D)
        neg_e = E_item[neg_idx]    # (B, D)

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)

        mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on user/pos/neg embeddings
        reg_loss = (
            u_e.norm(2).pow(2)
            + pos_e.norm(2).pow(2)
            + neg_e.norm(2).pow(2)
        ) / user_idx.size(0)

        loss = mf_loss + self.l2_reg * reg_loss
        return loss

    # ------------------------------------------------------------------
    # Contrastive part
    # ------------------------------------------------------------------
    def contrastive_loss(
        self,
        train_user_pos: dict,
        train_item_pos: dict,
        num_user_pairs: int = 1024,
        num_item_pairs: int = 1024,
    ):
        """
        Contrastive loss on layer-0 embeddings, approximated with random sampling.

        Parameters
        ----------
        train_user_pos : dict[int, set[int]]
            Mapping user -> set of interacted items (train only).
        train_item_pos : dict[int, set[int]]
            Mapping item -> set of interacted users (train only).
        num_user_pairs : int
            Number of user-user pairs to sample.
        num_item_pairs : int
            Number of item-item pairs to sample.

        Returns
        -------
        L_u : torch.Tensor
            User-side contrastive loss (scalar).
        L_i : torch.Tensor
            Item-side contrastive loss (scalar).
        """
        device = self.embedding.weight.device

        # Use layer-0 embeddings as in LightKG (E^{(0)})
        E0 = self.embedding.weight   # (N, D)
        E0_user = E0[: self.user_num]                                    # (U, D)
        E0_item = E0[self.user_num : self.user_num + self.item_num]      # (I, D)

        # -------------------------------
        # User-side contrastive loss L_u
        # -------------------------------
        user_ids = list(train_user_pos.keys())
        if len(user_ids) < 2:
            L_u = torch.zeros((), device=device)
        else:
            pairs_u = []
            for _ in range(num_user_pairs):
                u1, u2 = np.random.choice(user_ids, size=2, replace=False)
                pairs_u.append((u1, u2))

            s_list = []
            w_list = []
            u1_list = []
            u2_list = []

            for (u1, u2) in pairs_u:
                items1 = train_user_pos[u1]
                items2 = train_user_pos[u2]
                if len(items1) == 0 or len(items2) == 0:
                    continue

                inter = len(items1 & items2)
                deg1 = len(items1)
                deg2 = len(items2)

                denom = math.sqrt(deg1 * deg2)
                if denom == 0.0:
                    continue

                s = inter / denom if inter > 0 else 0.0   # similarity
                w = 1.0 - 1.0 / denom                     # degree-based weight

                u1_list.append(u1)
                u2_list.append(u2)
                s_list.append(s)
                w_list.append(w)

            if len(u1_list) == 0:
                L_u = torch.zeros((), device=device)
            else:
                u1_tensor = torch.tensor(u1_list, dtype=torch.long, device=device)
                u2_tensor = torch.tensor(u2_list, dtype=torch.long, device=device)
                s_tensor = torch.tensor(s_list, dtype=torch.float32, device=device)
                w_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)

                e1 = E0_user[u1_tensor]          # (M, D)
                e2 = E0_user[u2_tensor]          # (M, D)
                sim = (e1 * e2).sum(dim=1)       # (M,)

                # L_u ~ sum w * exp( (1 - s) * sim )
                L_u = (w_tensor * torch.exp((1.0 - s_tensor) * sim)).mean()

        # -------------------------------
        # Item-side contrastive loss L_i
        # -------------------------------
        item_ids = list(train_item_pos.keys())
        if len(item_ids) < 2:
            L_i = torch.zeros((), device=device)
        else:
            pairs_i = []
            for _ in range(num_item_pairs):
                i1, i2 = np.random.choice(item_ids, size=2, replace=False)
                pairs_i.append((i1, i2))

            s_list = []
            w_list = []
            i1_list = []
            i2_list = []

            for (i1, i2) in pairs_i:
                users1 = train_item_pos[i1]
                users2 = train_item_pos[i2]
                if len(users1) == 0 or len(users2) == 0:
                    continue

                inter = len(users1 & users2)
                deg1 = len(users1)
                deg2 = len(users2)

                denom = math.sqrt(deg1 * deg2)
                if denom == 0.0:
                    continue

                s = inter / denom if inter > 0 else 0.0
                w = 1.0 - 1.0 / denom

                i1_list.append(i1)
                i2_list.append(i2)
                s_list.append(s)
                w_list.append(w)

            if len(i1_list) == 0:
                L_i = torch.zeros((), device=device)
            else:
                i1_tensor = torch.tensor(i1_list, dtype=torch.long, device=device)
                i2_tensor = torch.tensor(i2_list, dtype=torch.long, device=device)
                s_tensor = torch.tensor(s_list, dtype=torch.float32, device=device)
                w_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)

                e1 = E0_item[i1_tensor]          # (M, D)
                e2 = E0_item[i2_tensor]          # (M, D)
                sim = (e1 * e2).sum(dim=1)       # (M,)

                L_i = (w_tensor * torch.exp((1.0 - s_tensor) * sim)).mean()

        return L_u, L_i

    # ------------------------------------------------------------------
    # Total loss: BPR + contrastive
    # ------------------------------------------------------------------
    def total_loss(
        self,
        user_idx: torch.LongTensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor,
        train_user_pos: dict,
        train_item_pos: dict,
        num_user_pairs: int = 1024,
        num_item_pairs: int = 1024,
    ):
        """
        Total loss = BPR + beta_user * L_u + beta_item * L_i.
        """
        bpr = self.bpr_loss(user_idx, pos_idx, neg_idx)
        L_u, L_i = self.contrastive_loss(
            train_user_pos=train_user_pos,
            train_item_pos=train_item_pos,
            num_user_pairs=num_user_pairs,
            num_item_pairs=num_item_pairs,
        )

        loss = bpr + self.beta_user * L_u + self.beta_item * L_i
        return loss #, bpr.detach(), L_u.detach(), L_i.detach()


def evaluator(
    model,
    data_loader,
    k: int,
    device,
    num_neg: int = 100,
    user_sample_size: int = 10000,
):
    """
    Evaluate LightKG model with leave-one-out style ranking:
    - For each user with at least one test interaction:
      - Choose one test item as the target.
      - Sample `num_neg` negative items.
      - Rank target among negatives and compute NDCG@k, Hit@k.

    This version:
    - Calls model.propagate() ONCE to get all user/item embeddings.
    - Then uses dot product for scoring.

    Returns
    -------
    ndcg : float
    hit  : float
    """
    user_num, item_num = data_loader.user_num, data_loader.item_num

    train_user_pos = data_loader.train_user_pos  # dict: user -> set(items)
    test_user_pos = data_loader.test_user_pos    # dict: user -> set(items)

    all_users = list(test_user_pos.keys())
    if len(all_users) == 0:
        return 0.0, 0.0

    if len(all_users) > user_sample_size:
        users = np.random.choice(all_users, size=user_sample_size, replace=False)
    else:
        users = all_users

    model.eval()
    with torch.no_grad():
        # Get final embeddings once
        E_user, E_item = model.propagate()  # (U, D), (I, D)
        # Ensure on CPU numpy for ranking
        E_user = E_user.detach().cpu()
        E_item = E_item.detach().cpu()

        NDCG = 0.0
        HIT = 0.0
        valid_user = 0

        all_items = np.arange(item_num)

        for u in users:
            test_items = list(test_user_pos.get(u, []))
            if len(test_items) == 0:
                continue

            # pick one target test item
            target = np.random.choice(test_items)

            # items already interacted with (train + other test)
            rated = set(train_user_pos.get(u, set()))
            rated.update(test_items)

            # sample negatives
            neg_items = []
            while len(neg_items) < num_neg:
                j = np.random.randint(0, item_num)
                if j not in rated and j not in neg_items:
                    neg_items.append(j)

            # candidate items: [target] + negatives
            item_idx = np.array([target] + neg_items, dtype=np.int64)

            # get embeddings
            u_emb = E_user[u].unsqueeze(0)             # (1, D)
            i_emb = E_item[item_idx]                   # (1 + num_neg, D)

            # scores = dot(u, i)
            scores = (u_emb * i_emb).sum(dim=1).numpy()  # (1 + num_neg,)

            # rank target (index 0)
            rank = (-scores).argsort().tolist().index(0)

            valid_user += 1
            if rank < k:
                HIT += 1
                NDCG += 1 / np.log2(rank + 2)

    if valid_user == 0:
        return 0.0, 0.0

    ndcg = NDCG / valid_user
    hit = HIT / valid_user
    return ndcg, hit



def trainer(
    model,
    data_loader,
    optimizer,
    batch_size: int,
    epoch_num: int,
    num_batches_per_epoch: int,
    eval_interval: int,
    eval_k: int,
    patience: int,
    best_model_path: str,
    device,
    num_user_pairs: int = 1024,
    num_item_pairs: int = 1024,
):
    """
    Training loop for LightKG with:
      - BPR loss
      - Contrastive losses L_u, L_i (on layer-0 embeddings)
      - Early stopping based on NDCG@k on eval set.
    """

    best_ndcg = float("-inf")
    best_hit = 0.0
    epochs_without_improve = 0

    train_user_pos = data_loader.train_user_pos    # dict: u -> set(items)
    train_item_pos = data_loader.train_item_pos    # dict: i -> set(users)

    for epoch in range(1, epoch_num + 1):
        model.train()
        epoch_loss = 0.0
        epoch_bpr = 0.0
        epoch_Lu = 0.0
        epoch_Li = 0.0

        for _ in range(num_batches_per_epoch):
            users, pos_items, neg_items = data_loader.get_bpr_batch(batch_size)

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()

            # Total loss = BPR + beta_u * L_u + beta_i * L_i
            loss, bpr, L_u, L_i = model.total_loss(
                user_idx=users,
                pos_idx=pos_items,
                neg_idx=neg_items,
                train_user_pos=train_user_pos,
                train_item_pos=train_item_pos,
                num_user_pairs=num_user_pairs,
                num_item_pairs=num_item_pairs,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bpr += bpr.item()
            epoch_Lu += L_u.item()
            epoch_Li += L_i.item()

        avg_loss = epoch_loss / num_batches_per_epoch
        avg_bpr = epoch_bpr / num_batches_per_epoch
        avg_Lu = epoch_Lu / num_batches_per_epoch
        avg_Li = epoch_Li / num_batches_per_epoch

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {avg_loss:.4f} "
            f"(BPR: {avg_bpr:.4f}, Lu: {avg_Lu:.4f}, Li: {avg_Li:.4f})"
        )

        # evaluate & early stopping
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                ndcg, hit = evaluator(
                    model=model,
                    data_loader=data_loader,
                    k=eval_k,
                    device=device,
                )

            print(f"Eval - NDCG@{eval_k}: {ndcg:.4f}, Hit@{eval_k}: {hit:.4f}")

            # check improvement
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_hit = hit

                torch.save(model.state_dict(), best_model_path)
                print(f"  ** Best model updated and saved to '{best_model_path}' **")

                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                print(f"  No improvement. Patience: {epochs_without_improve}/{patience}")

                if epochs_without_improve >= patience:
                    print("  >>> Early stopping triggered.")
                    break

    print("========================================")
    print(f"Best Eval : NDCG@{eval_k}={best_ndcg:.4f}, Hit@{eval_k}={best_hit:.4f}")
    print(f"Best model weights saved at: {best_model_path}")
