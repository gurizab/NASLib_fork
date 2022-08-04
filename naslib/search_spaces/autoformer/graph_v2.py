import os
import pickle
import numpy as np
import random
import itertools
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive, Identity
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
)
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from naslib.utils.utils import iter_flatten, AttrDict
from naslib.search_spaces.autoformer.model.module.preprocess import Preprocess, Preprocess_partial
from naslib.utils.utils import get_project_root
from naslib.search_spaces.autoformer.model.module.primitives import Stack, Split, Super_embed_head_choice_qkv
from naslib.search_spaces.autoformer.model.module.primitives import qkv_super
from naslib.search_spaces.autoformer.model.module.primitives import QKV_super_head_choice, LinearEmb, \
    Super_embed_choice_linear, Super_embed_choice_qkv, Dropout_emb_choice, RelativePosition2D_super, Proj_emb_choice, \
    Dropout, AttnFfnNorm_embed_choice, Scale, \
    LinearSuper_Emb_Ratio_Combi, Norm_embed_choice, Proj_head_emb_choice, Super_embed_head_choice_qkv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from naslib.search_spaces.autoformer.model.module.Linear_super import LinearSuper
from naslib.search_spaces.autoformer.model.module.layernorm_super import LayerNormSuper
from naslib.search_spaces.autoformer.model.module.multihead_super import AttentionSuper
from naslib.search_spaces.autoformer.model.utils import trunc_normal_
from naslib.search_spaces.autoformer.model.utils import DropPath
import numpy as np


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AutoformerSearchSpace(Graph):
    """
    Implementation of the AutoFormer search space.
    """

    QUERYABLE = False

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 in_chans=3,
                 num_classes=10,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pre_norm=True,
                 scale=True,
                 gp=True,
                 relative_position=True,
                 change_qkv=True,
                 abs_pos=True,
                 max_relative_position=14):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self,
                                                       "NUM_CLASSES") else 10
        self.op_indices = None
        self.pre_norm = pre_norm
        self.max_epoch = 199
        self.space_name = "autoformer"
        self.num_classes = num_classes
        self.name = "makrograph"
        self.scale = scale
        self.drop_path_rate = drop_path_rate
        self.relative_position = relative_position
        self.attn_drop_rate = attn_drop_rate
        self.qk_scale = qk_scale
        # Cell is on the edges
        # 1-2:               Preprocessing
        # 2-3, ..., 6-7:     cells stage 1
        # 7-8:               residual block stride 2
        # 8-9, ..., 12-13:   cells stage 2
        # 13-14:             residual block stride 2
        # 14-15, ..., 18-19: cells stage 3
        # 19-20:             post-processing
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.block = 0

        #
        self.choices = {
            'num_heads': [3, 4],
            'mlp_ratio': [3.5, 4],
            'embed_dim': [192, 216, 240],
            'depth': [12, 13, 14]
        }
        # operations at the edges
        #

        self.change_qkv = change_qkv
        self.scale = scale
        self.depth_super = max(self.choices["depth"])
        self.super_num_heads = max(self.choices["num_heads"])
        self.super_head_dim = self.super_num_heads * 64 * 3
        self.super_embed_dim = max(self.choices["embed_dim"])
        self.super_mlp_ratio = max(self.choices["mlp_ratio"])
        self.super_ffn_embed_dim_this_layer = int(
            max(self.choices["mlp_ratio"]) * self.super_embed_dim)
        self.total_num_nodes = 1 + 4 + 16 * self.depth_super
        print(self.total_num_nodes)
        self.add_nodes_from(range(1, self.total_num_nodes))
        self.add_edges_from([(i, i + 1)
                             for i in range(1, self.total_num_nodes)])

        self.groups = {"embed_dim": len(self.choices["embed_dim"])}
        for i in self.depth_super:
            self.groups["num_heads_" + str(i)] = len(self.choices["num_heads"])
            self.groups["mlp_ratio_" + str(i)] = len(self.choices["mlp_ratio"])

        self.preprocess_super = Preprocess(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim_list=self.choices["embed_dim"],
            abs_pos=abs_pos,
            pre_norm=pre_norm)
        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # preprocessing
        num_patches = self.preprocess_super.num_patches
        self.patch_emb_op_list = []
        self.pre_norm = pre_norm
        for e in self.choices["embed_dim"]:
            self.patch_emb_op_list.append(
                Preprocess_partial(self.preprocess_super, e))
        self.edges[1, 2].set("op", self.patch_emb_op_list)
        self.edges[1, 2].set("group", "embed_dim")
        start = 2
        # for i in range(self.depth_super):

        # set comb_op and split
        self.nodes[self.total_num_nodes - 3]["comb_op"] = Stack()
        self.depth_choice_list = [
            Split(idx) for idx in range(len(self.choices["depth"]))
        ]
        self.edges[self.total_num_nodes - 3, self.total_num_nodes - 2].set(
            'op', self.depth_choice_list)
        """self.edges[self.total_num_nodes - 3, self.total_num_nodes - 2].set(
            "group", "depth")"""

        start = 2
        self.ffn_layer_norm_list = torch.nn.ModuleList()
        self.attn_layer_norm_list = torch.nn.ModuleList()
        self.qkv_super_list = torch.nn.ModuleList()
        self.rel_pos_embed_k_list = torch.nn.ModuleList()
        self.rel_pos_embed_v_list = torch.nn.ModuleList()
        self.proj_list = torch.nn.ModuleList()
        self.proj_drop_list = torch.nn.ModuleList()
        self.fc1_list = torch.nn.ModuleList()
        self.fc2_list = torch.nn.ModuleList()
        for i in range(self.depth_super):
            self.ffn_layer_norm_list.append(
                LayerNormSuper(self.super_embed_dim))
            self.attn_layer_norm_list.append(
                LayerNormSuper(self.super_embed_dim))
            if change_qkv:
                self.qkv_super_list.append(
                    qkv_super(self.super_embed_dim,
                              3 * 64 * max(self.choices["num_heads"]),
                              bias=qkv_bias))
            else:
                self.qkv_super_list.append(
                    LinearSuper(self.super_embed_dim,
                                3 * self.super_embed_dim,
                                bias=qkv_bias))
            if not change_qkv:
                self.rel_pos_embed_k_list.append(
                    RelativePosition2D_super(self.super_embed_dim // self.super_num_heads, max_relative_position))
                self.rel_pos_embed_v_list.append(
                    RelativePosition2D_super(self.super_embed_dim // self.super_num_heads, max_relative_position))
            else:
                self.rel_pos_embed_k_list.append(
                    RelativePosition2D_super(64, max_relative_position))
                self.rel_pos_embed_v_list.append(RelativePosition2D_super(64, max_relative_position))
            self.proj_list.append(
                LinearSuper(self.super_embed_dim, self.super_embed_dim))
            self.proj_drop_list.append(Dropout(drop_rate))
            self.fc1_list.append(
                LinearSuper(super_in_dim=self.super_embed_dim,
                            super_out_dim=self.super_ffn_embed_dim_this_layer))
            self.fc2_list.append(
                LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer,
                            super_out_dim=self.super_embed_dim))
        # for i in range(self.depth_super):
        #    cell = self.init_transformer_block(i)
        #    self.transformer_block_list.append(cell)
        start = 2
        # if i != self.depth_super - 1 and (i + 1 in self.choices["depth"]):
        #        self.add_edges_from([(start, self.total_num_nodes - 3)])
        #        self.edges[start, self.total_num_nodes - 3].set(
        #            "op", ops.Identity()
        #        )  # edges[start, start + 6].set("op", ops.Identity())
        self.qkv_embed_choice_list_blocks = []
        self.qkv_head_choice_list_blocks = []
        self.qkv_head_emb_choice_list_blocks = []
        self.proj_emb_choice_list_blocks = []
        self.dropout_emb_choice_list_blocks = []
        self.fc2_emb_r_choice_list_blocks = []
        self.attn_norm_choice_list_blocks = []
        self.ffn_norm_choice_list_blocks = []
        self.fc1_emb_r_choice_list_blocks = []
        self.dropout_emb_full_choice_list_blocks = []
        self.scale_choice_list_blocks = []
        self.ffn_norm_choice_list_after_blocks = []
        self.proj_emb_head_choice_list_blocks = []
        for i in range(self.depth_super):

            self.add_nodes_from(range(start, start + 18))
            self.add_edges_from([(start, start + 7)])
            self.add_edges_from([(start + 8, start + 15)])
            if not self.change_qkv:
                self.qkv_embed_choice_list = []

                for e in self.choices["embed_dim"]:
                    self.qkv_embed_choice_list.append(
                        Super_embed_choice_linear(self.qkv_super_list[i],
                                                  self.attn_layer_norm, e,
                                                  self.super_embed_dim,
                                                  self.pre_norm))
                self.edges[start, start + 1].set("op", self.qkv_embed_choice_list)
                self.edges[start, start + 1].set("group", "embed_dim")
                self.qkv_embed_choice_list_blocks.append(self.qkv_embed_choice_list)
                self.edges[start + 1, start + 2].set("op", ops.Identity())
                self.edges[start + 1, start + 2].finalize()
            else:
                self.qkv_embed_choice_list = []

                for e in self.choices["embed_dim"]:
                    self.qkv_embed_choice_list.append(
                        Super_embed_choice_qkv(self.attn_layer_norm, e,
                                               self.super_embed_dim,
                                               self.pre_norm))
                self.edges[start, start + 1].set("op", self.qkv_embed_choice_list)
                self.edges[start, start + 1].set("group", "embed_dim")
                self.qkv_embed_choice_list_blocks.append(self.qkv_embed_choice_list)
                self.qkv_head_emb_choice_list = []
                for e in self.choices["embed_dim"]:
                    for h in self.choices["num_heads"]:
                        self.qkv_head_emb_choice_list.append(
                            Super_embed_head_choice_qkv(
                                self.qkv_super_list[i], e, h, self.super_head_dim))
                self.edges[start + 1, start + 2].set("op", self.qkv_head_emb_choice_list)
                self.edges[start + 1, start + 2].set("group", ["embed_dim", "num_heads_" + str(i)])
                self.edges[start + 1, start + 2].set("mixed_op_type", "cross_op")
                self.qkv_head_emb_choice_list_blocks.append(self.qkv_head_emb_choice_list)

            self.qkv_head_choice_list = []
            for h in self.choices["num_heads"]:
                self.qkv_head_choice_list.append(
                    QKV_super_head_choice(self.rel_pos_embed_k_list[i],
                                          self.rel_pos_embed_v_list[i], h,
                                          self.attn_drop_rate, self.super_embed_dim,
                                          self.super_head_dim, self.change_qkv,
                                          self.relative_position, self.scale, self.super_num_heads))
            self.edges[start + 2, start + 3].set("op", self.qkv_head_choice_list)
            self.edges[start + 2, start + 3].set("group", "num_heads_" + str(i))
            self.qkv_head_choice_list_blocks.append(self.qkv_head_choice_list)

            if not change_qkv:
                self.proj_emb_choice_list = []
                for e in self.choices["embed_dim"]:
                    self.proj_emb_choice_list.append(Proj_emb_choice(self.proj_list[i], e, self.super_embed_dim))
                self.edges[start + 3, start + 4].set("op", self.proj_emb_choice_list)
                self.edges[start + 3, start + 4].set("group", "embed_dim")
                self.proj_emb_choice_list_blocks.append(self.proj_emb_choice_list)

            else:
                self.proj_head_emb_choice_list = []
                for e in self.choices["embed_dim"]:
                    for h in self.choices["num_heads"]:
                        self.proj_head_emb_choice_list.append(
                            Proj_head_emb_choice(self.proj_list[i], e, h, self.super_embed_dim))
                self.edges[start + 3, start + 4].set("op", self.proj_head_emb_choice_list)
                self.edges[start + 3, start + 4].set("group", ["embed_dim", "num_heads_" + str(i)])
                self.edges[start + 3, start + 4].set("mixed_op_type", "cross_op")
                self.proj_emb_head_choice_list_blocks.append(self.proj_head_emb_choice_list)
            self.edges[start + 4, start + 5].set("op", self.proj_drop_list[i])
            self.edges[start + 4, start + 5].finalize()

            self.dropout_emb_choice_list = []
            for e in self.choices["embed_dim"]:
                self.dropout_emb_choice_list.append(
                    Dropout_emb_choice(e, self.super_attn_dropout,
                                       self.super_embed_dim))
            self.edges[start + 5, start + 6].set("op",
                                                 self.dropout_emb_choice_list)
            self.edges[start + 5, start + 6].set("group", "embed_dim")
            self.dropout_emb_choice_list_blocks.append(self.dropout_emb_choice_list)
            dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth_super)]
            self.drop_path = DropPath(dpr[0]) if dpr[0] > 0. else ops.Identity()
            self.edges[start + 6, start + 7].set("op", self.drop_path)
            self.edges[start + 6, start + 7].finalize()
            self.edges[start, start + 7].set("op", ops.Identity())
            self.edges[start, start + 7].finalize()

            self.attn_norm_choice_list = []
            for e in self.choices["embed_dim"]:
                self.attn_norm_choice_list.append(
                    AttnFfnNorm_embed_choice(self.attn_layer_norm_list[i],
                                             e,
                                             self.super_embed_dim,
                                             self.pre_norm,
                                             after=True))
            self.edges[start + 7, start + 8].set("op", self.attn_norm_choice_list)
            self.edges[start + 7, start + 8].set("group", "embed_dim")
            self.attn_norm_choice_list_blocks.append(self.attn_norm_choice_list)

            self.ffn_norm_choice_list = []
            for e in self.choices["embed_dim"]:
                self.ffn_norm_choice_list.append(
                    AttnFfnNorm_embed_choice(self.ffn_layer_norm_list[i],
                                             e,
                                             self.super_embed_dim,
                                             self.pre_norm,
                                             before=True))
            self.edges[start + 8, start + 9].set("op", self.ffn_norm_choice_list)
            self.edges[start + 8, start + 9].set("group", "embed_dim")
            self.ffn_norm_choice_list_blocks.append(self.ffn_norm_choice_list)

            self.fc1 = self.fc1_list[i]
            self.fc2 = self.fc2_list[i]
            self.fc1_emb_r_choice_list = []
            for e in self.choices["embed_dim"]:
                for r in self.choices["mlp_ratio"]:
                    self.fc1_emb_r_choice_list.append(
                        LinearSuper_Emb_Ratio_Combi(
                            self.fc1, self.super_ffn_embed_dim_this_layer,
                            self.super_embed_dim, self.super_mlp_ratio, e, r))
            self.edges[start + 9, start + 10].set("op", self.fc1_emb_r_choice_list)
            self.edges[start + 9, start + 10].set("group", ["embed_dim", "mlp_ratio_"+str(i)])
            self.edges[start + 9, start + 10].set("mixed_op_type", "cross_op")
            self.fc1_emb_r_choice_list_blocks.append(self.fc1_emb_r_choice_list)

            self.dropout_emb_full_choice_list = []
            for e in self.choices["embed_dim"]:
                self.dropout_emb_full_choice_list.append(
                    Dropout_emb_choice(e, self.super_dropout,
                                       self.super_embed_dim))
            self.edges[start + 10, start + 11].set("op", self.dropout_emb_full_choice_list)
            self.edges[start + 10, start + 11].set("group", "embed_dim")
            self.dropout_emb_full_choice_list_blocks.append(self.dropout_emb_full_choice_list)
            self.fc2_emb_r_choice_list = []
            for e in self.choices["embed_dim"]:
                for r in self.choices["mlp_ratio"]:
                    self.fc2_emb_r_choice_list.append(
                        LinearSuper_Emb_Ratio_Combi(
                            self.fc2,
                            self.super_ffn_embed_dim_this_layer,
                            self.super_embed_dim,
                            self.super_mlp_ratio,
                            e,
                            r,
                            reverse=True,
                            scale=True))
            self.edges[start + 11, start + 12].set("op",
                                                   self.fc2_emb_r_choice_list)
            self.edges[start + 11, start + 12].set("group", ["embed_dim", "mlp_ratio_"+str(i)])
            self.edges[start + 11, start + 12].set("mixed_op_type", "cross_op")
            self.fc2_emb_r_choice_list_blocks.append(self.fc2_emb_r_choice_list)

            self.edges[start + 12, start + 13].set(
                "op", self.dropout_emb_full_choice_list)
            self.edges[start + 12, start + 13].set("group", "embed_dim")
            self.dropout_emb_full_choice_list_blocks.append(self.dropout_emb_full_choice_list)

            self.scale_choice_list = []
            for r in self.choices["mlp_ratio"]:
                self.scale_choice_list.append(
                    Scale(self.super_mlp_ratio, self.super_embed_dim, r))
            self.edges[start + 13, start + 14].set("group", "mlp_ratio_" + str(i))
            if self.scale == True:
                self.edges[start + 13, start + 14].set("op",
                                                       self.scale_choice_list)
                self.scale_choice_list_blocks.append(self.scale_choice_list)
            else:
                self.edges[start + 13, start + 14].set("op", ops.Identity())
                self.edges[start + 13, start + 14].finalize()

            self.edges[start + 14, start + 15].set("op", self.drop_path)
            self.edges[start + 14, start + 15].finalize()
            self.edges[start + 8, start + 15].set("op", ops.Identity())
            self.edges[start + 8, start + 15].finalize()
            self.ffn_norm_choice_list_after = []
            for e in self.choices["embed_dim"]:
                self.ffn_norm_choice_list_after.append(
                    AttnFfnNorm_embed_choice(self.ffn_layer_norm,
                                             e,
                                             self.super_embed_dim,
                                             self.pre_norm,
                                             after=True))
            self.edges[start + 15, start + 16].set("op",
                                                   self.ffn_norm_choice_list_after)
            self.edges[start + 15, start + 16].set("group", "embed_dim")
            self.ffn_norm_choice_list_after_blocks.append(self.ffn_norm_choice_list_after)
            # self.edges[start, start+1].finalize()
            start = start + 16
            if i != self.depth_super - 1 and (i + 1 in self.choices["depth"]):
                self.add_edges_from([(start - 1, self.total_num_nodes - 3)])
                self.edges[start - 1, self.total_num_nodes - 3].set(
                    "op", ops.Identity())
                self.edges[start - 1, self.total_num_nodes - 3].finalize()
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=self.super_embed_dim)
            self.norm_choice_list = []
            for e in self.choices["embed_dim"]:  # G2
                self.norm_choice_list.append(
                    Norm_embed_choice(self.norm, e, self.super_embed_dim, gp))
            self.edges[start + 1, start + 2].set("op", self.norm_choice_list)
            self.edges[start + 1, start + 2].set("group", "embed_dim")
        else:
            self.edges[start + 1, start + 2].set("op", ops.Identity())
            self.edges[start + 1, start + 2].set("group", "emnbed_dim")
            self.edges[start + 1, start + 2].finalize()
        self.head = LinearSuper(
            self.super_embed_dim,
            num_classes) if num_classes > 0 else ops.Identity()
        self.classifier_head_choice_list = []
        if num_classes > 0:
            for e in self.choices["embed_dim"]:  # G2

                self.classifier_head_choice_list.append(
                    LinearEmb(self.head, e, num_classes))
            self.edges[start + 2, start + 3].set(
                "op", self.classifier_head_choice_list)
            self.edges[start + 2, start + 3].set("group", "embed_dim")
        else:
            self.edges[start + 2, start + 3].set("op", ops.Identity())
            self.edges[start + 2, start + 3].finalize()
        # for u, v, data in self.edges.data():
        #    print(u)
        #    print(v)
        #    print(data)
        #    if data==None:
        #        print("Found None data")
        # self._delete_flagged_edges()

    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        '''op_indices_depth = np.random.randint(3, size=(1))
        depth = self.choices["depth"][op_indices_depth[0]]
        op_indices_emb = np.random.randint(3, size=(1))
        op_indices_emb = [op_indices_emb[0] for _ in range(depth)]
        op_indices_head = np.random.randint(3, size=(depth))
        op_indices_ratio_emb = np.random.choice([x for x in range(3 * op_indices_emb[0], 3 + 3 * op_indices_emb[0])],
                                                size=(depth))
        op_indices_ratio = [x % 3 for x in list(op_indices_ratio_emb)]'''
        op_indices_depth = [0]  # np.random.randint(3, size=(1))
        depth = self.choices["depth"][op_indices_depth[0]]  # [0]]
        op_indices_emb = [0] * depth  # np.random.randint(3, size=(1))
        # op_indices_emb = [op_indices_emb[0] for _ in range(depth)]
        op_indices_head = [2] * depth  # np.random.randint(3, size=(depth))
        op_indices_head_emb = [
                                  3] * depth  # np.random.choice([x for x in range(3*op_indices_emb[0],3+3*op_indices_emb[0])], size=(depth))
        op_indices_ratio_emb = [
                                   3] * depth
        op_indices_ratio = [2] * depth  # [x%3 for x in list(op_indices_ratio_emb)]
        print("Choosing emp index", op_indices_emb)
        print("Choosing head indices", op_indices_head)
        print("Choosing mlp_ratio_op indices", op_indices_ratio)
        print("Depth", depth)
        self.set_op_indices(op_indices_depth, op_indices_emb, op_indices_head,
                            op_indices_ratio_emb, op_indices_head_emb, op_indices_ratio, depth)

    def set_op_indices(self, op_indices_depth, op_indices_emb, op_indices_head,
                       op_indices_ratio_emb, op_indices_head_emb, op_indices_ratio, depth):
        # This will update the edges in the naslib object to op_indices
        self.edges[1, 2].set("op", self.patch_emb_op_list[op_indices_emb[0]])
        start = 2
        for i in range(depth):
            # print(i)
            # cell = self.transformer_block_list[i]
            if not self.change_qkv:
                self.edges[start, start + 1].set("op", self.qkv_embed_choice_list_blocks[i][op_indices_emb[i]])
                self.edges[start + 3, start + 4].set("op", self.proj_emb_choice_list_blocks[i][op_indices_emb[i]])
            else:
                self.edges[start, start + 1].set("op", self.qkv_embed_choice_list_blocks[i][op_indices_emb[i]])
                self.edges[start + 1, start + 2].set("op",
                                                     self.qkv_head_emb_choice_list_blocks[i][op_indices_head_emb[i]])
                self.edges[start + 3, start + 4].set("op",
                                                     self.proj_emb_head_choice_list_blocks[i][op_indices_head_emb[i]])
            self.edges[start + 2, start + 3].set(
                "op", self.qkv_head_choice_list_blocks[i][op_indices_head[i]])
            self.edges[start + 5, start + 6].set(
                "op", self.dropout_emb_choice_list_blocks[i][op_indices_emb[i]])
            self.edges[start + 7, start + 8].set(
                "op", self.attn_norm_choice_list_blocks[i][op_indices_emb[i]])
            self.edges[start + 8, start + 9].set(
                "op", self.ffn_norm_choice_list_blocks[i][op_indices_emb[i]])
            self.edges[start + 9, start + 10].set(
                "op", self.fc1_emb_r_choice_list_blocks[i][op_indices_ratio_emb[i]])
            self.edges[start + 10, start + 11].set(
                "op", self.dropout_emb_full_choice_list_blocks[i][op_indices_emb[i]])
            self.edges[start + 11, start + 12].set(
                "op", self.fc2_emb_r_choice_list_blocks[i][op_indices_ratio_emb[i]])
            self.edges[start + 12, start + 13].set(
                "op", self.dropout_emb_full_choice_list_blocks[i][op_indices_emb[i]])
            if self.scale == True:
                self.edges[start + 13, start + 14].set(
                    "op", self.scale_choice_list_blocks[i][op_indices_ratio[i]])
            self.edges[start + 15, start + 16].set(
                "op", self.ffn_norm_choice_list_after_blocks[i][op_indices_emb[i]])
            start = start + 16
        print(start)
        print(self.total_num_nodes - 3)
        for i in range(start, self.total_num_nodes - 3):
            self.edges[i, i + 1].set("op", ops.Identity())
            # self.edges[i, i + 1].finalize()
        # print(self.edges[self.total_num_nodes - 3, self.total_num_nodes - 2])
        start = self.total_num_nodes - 3
        self.edges[start, start + 1].set(
            "op", self.depth_choice_list[op_indices_depth[0]])
        if self.pre_norm:
            self.edges[start + 1, start + 2].set(
                "op", self.norm_choice_list[op_indices_emb[-1]])
        if self.num_classes > 0:
            self.edges[start + 2, start + 3].set(
                "op", self.classifier_head_choice_list[op_indices_emb[-1]])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def count_parameters_in_MB(model):
    return np.sum(
        np.prod(v.size()) for name, v in model.named_parameters()
        if "auxiliary" not in name) / 1e6


# ss = AutoformerSearchSpace(num_classes=10)

'''import networkx as nx
nx.draw(ss, with_labels=True, pos=nx.kamada_kawai_layout(ss))
plt.show()
plt.savefig('autoformer.png')
ss.sample_random_architecture()
ss.parse()
x = torch.randn([2,3,32,32])#.cuda()
out = ss(x)
import copy
g = copy.deepcopy(ss)
print(out.shape)'''
'''
loss = torch.sum(out)
loss.backward()
for k, v in ss.named_parameters():
    name = k
    param = v
    if param.grad==None:
        print(name)
    else:
     if torch.sum(param.grad)==0:
       print(name)
       #print(param.grad)
graph = ss.clone().unparse()
for k, v in graph.named_parameters():
    name = k
    param = v
    print(name)



'''

# ss.parse()
'''ss = AutoformerSearchSpace(num_classes=10)
import networkx as nx

nx.draw(ss, with_labels=True, pos=nx.kamada_kawai_layout(ss))
plt.show()
plt.savefig('autoformer.png')
print(ss._get_child_graphs(single_instances=False))
for graph in ss._get_child_graphs(single_instances=False) + [ss]:
            print("Graph name", graph.name)
            for u, v, edge_data in graph.edges.data():
                print(edge_data)
'''

'''ss.sample_random_architecture()
ss.parse()
ss = ss.to("cuda")
optim = torch.optim.AdamW(ss.parameters(), lr=5e-4)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

writer = SummaryWriter('test_patch_2_naslib')
step = 0
running_loss = 0
#for k, v in ss.named_parameters():
#    name = k
#    param = torch.nn.Parameter(v)
#    print(name)
#    param.data.fill_(1)
#input = torch.ones([2, 3, 32, 32])
#ss.eval()
#print(ss(input))
#print(ss.modules_str())
#ss = torch.nn.DataParallel(ss)
#ss.to("cuda")
for i in range(200):
    #print(ss.config)
    print("starting epoch", i)
    for i, data in enumerate(trainloader, 0):
        ss.train()
        step = step + 1
        inputs, targets = data
        optim.zero_grad()
        inputs = inputs.to("cuda")
        writer.add_graph(ss, inputs)
        loss_fn = torch.nn.CrossEntropyLoss()
        out = ss(inputs)
        #print("Out", torch.argmax(out, dim=-1))
        #print("Targets", targets)

        loss = loss_fn(out, targets.cuda())
        #print("loss", loss)
        loss.backward()
        # for name, param in ss.named_parameters():
        #    print(name)
        #    print(param.grad)
        optim.step()
        writer.add_scalar('training loss', loss, step)
        #break
    ss.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our output
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to("cuda"), labels.to("cuda")
            # calculate outputs by running images through the network
            outputs = ss(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        #break
    #break
writer.close()'''
