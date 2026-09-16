"""
精排单塔打分模型（RankModel）—— 故意埋两个性能坑，供性能分析学习用。

精排场景：一个用户 + N 个候选 item，逐 item 打分（点击率）。

【坑2 —— user 特征冗余 N 次】
    user_tower 处理用户历史序列（含明确 GEMM 耗时），但它放在候选 item 循环里，
    每个 item 都重新算一遍。由于 user_tower 的结果对所有候选 item 是完全相同的，
    这里被白白重复计算了 N 次（N = num_candidates）。
    → 正确做法：user_tower 提出循环外，只算一次，缓存复用。
    → 在 Perfetto/nsys 里你能看到：user_tower 区间（或其中的 GEMM kernel）重复出现 N 次。

【坑1 —— launch overhead（kernel 碎片化）】
    item_tower 里故意塞了 n_small_ops 个独立的微小 element-wise 算子（x = x*w+b; relu），
    每个 kernel 在 GPU 上只执行 1~2us，但 CPU 下发 + Python 调度的 launch 开销要大得多，
    于是 GPU 时间轴上这些 kernel 之间出现大量空隙（GPU 空等 CPU 喂 kernel）。
    → 正确做法：把几十个小算子融合成几个大 kernel，或上 CUDA Graph。
    → 在 Perfetto/nsys 里你能看到：item_tower 区间内一堆微秒级小 kernel + 明显空隙。
"""
import torch
import torch.nn as nn

# NVTX（供 nsys 区分 user_tower / item_tower 两个区间）
try:
    import torch.cuda.nvtx as _nvtx
    _nvtx.range_push("nvtx_probe")
    _nvtx.range_pop()
    HAS_NVTX = True
except Exception:
    _nvtx = None
    HAS_NVTX = False


def _push(name):
    if HAS_NVTX:
        _nvtx.range_push(name)


def _pop():
    if HAS_NVTX:
        _nvtx.range_pop()


class RankModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ed = config['emb_dim']
        self.num_users = config['num_users']
        self.num_items = config['num_items']

        self.user_id_emb = nn.Embedding(self.num_users, ed)
        self.item_emb = nn.Embedding(self.num_items, ed)
        self.hist_emb = nn.Embedding(self.num_items, ed)   # 历史序列里也是 item id

        # ---- 坑2 载体：user tower（有明确 GEMM 耗时，会在 item 循环里重复）----
        up = config['user_proj_dim']
        self.user_proj = nn.Linear(ed, up)                  # [L, ed] -> [L, up]（M=L 的 GEMM）
        layers = []
        in_d = ed + up   # user_id embedding + 历史序列池化 拼接
        for h in config['user_hidden']:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        self.user_mlp = nn.Sequential(*layers)

        # ---- item 侧投影 ----
        ip = config['item_proj_dim']
        self.item_proj = nn.Linear(ed, ip)

        # ---- 坑1 载体：item tower 内的大量微小算子 ----
        n = config['n_small_ops']
        self.small_w = nn.Parameter(torch.ones(n))
        self.small_b = nn.Parameter(torch.zeros(n))

        # ---- score head ----
        user_out_dim = config['user_hidden'][-1]
        self.score_head = nn.Sequential(
            nn.Linear(user_out_dim + ip, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def user_tower(self, user_id, hist):
        """用户侧表示：user_id embedding + 历史序列池化 -> MLP。

        注意：这个函数对同一用户的结果是固定的，但 forward 里被重复调用 N 次（坑2）。
        """
        u_id = self.user_id_emb(user_id)        # [ed]
        x = torch.relu(self.user_proj(hist))    # [L, up]
        x = x.mean(dim=0)                        # [up]
        x = torch.cat([u_id, x])                 # [ed + up]
        x = self.user_mlp(x)                     # [user_out_dim]
        return x

    def item_tower(self, item_vec):
        """item 侧：投影 + 大量微小算子（坑1）。"""
        x = self.item_proj(item_vec)            # [ip]
        # 故意塞 n_small_ops 个独立微小 kernel，制造 launch overhead
        for i in range(self.small_w.shape[0]):
            x = x * self.small_w[i] + self.small_b[i]
            x = torch.relu(x)
        return x

    def forward(self, user_id, hist_ids, item_ids):
        """
        user_id : 标量（0-d tensor）
        hist_ids: [hist_len] 用户历史行为序列
        item_ids: [N] 候选 item
        返回 scores: [N]
        """
        hist = self.hist_emb(hist_ids)          # [L, ed]

        scores = []
        for i in range(item_ids.shape[0]):
            # ★坑2：user_tower 对每个候选 item 都重算一遍（结果其实完全相同）
            _push("user_tower")
            u = self.user_tower(user_id, hist)
            _pop()

            # ★坑1：item_tower 内有大量微小 kernel，制造 launch overhead
            _push("item_tower")
            v = self.item_tower(self.item_emb(item_ids[i]))
            _pop()

            _push("score_head")
            s = self.score_head(torch.cat([u, v]))
            _pop()
            scores.append(s)

        return torch.stack(scores).squeeze(-1)   # [N]
