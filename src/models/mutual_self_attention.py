# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange

from src.models.attention import TemporalBasicTransformerBlock

from .attention import BasicTransformerBlock


def torch_dfs(model: torch.nn.Module):      #递归遍历网络的所有层
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cuda"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )
        # hacked_basic_transformer_inner_forward是对diffusers库中attention.py的BasicTransformerBlock类的forward函数重写
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            audio_cond_fea: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
            audio_feature_ratio = 3.0
        ):
            # 判断使用哪种norm1
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:   # False
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:                                           # 进了这里
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )                           #得到的cross_attention_kwargs是一个字典,全程打印出来是一个空字典
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":     #如果是写模式，将经过norm的特征存进self.bank,然后进行一次自注意力
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    bank_feas = [           #读取模式中先将self.bank中的每个特征repeat到视频帧长度
                        rearrange(
                            d.unsqueeze(1).repeat(1, video_length, 1, 1),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]
                    modify_norm_hidden_states = torch.cat(          #将本次输入的特征与bank中的所有向量cat到一起
                        [norm_hidden_states] + bank_feas, dim=1
                    )
                    # print(f"modify_norm_hidden_states:{modify_norm_hidden_states.shape}")

                    hidden_states_uc = (            #进行一次残差自注意力
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    if do_classifier_free_guidance:             # 分类引导
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        # print(hidden_states_c.shape, _uc_mask.shape)
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        # print(hidden_states_c.shape, norm_hidden_states.shape, hidden_states.shape, _uc_mask.shape)
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask], # B * 4096 * 768
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:                  #进行交叉注意力
                        # Ref Cross-Attention
                        norm_hidden_states = (                  #先进行一次norm
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        # print("Audio Cross-Attention shapes:", norm_hidden_states.shape, audio_cond_fea.shape)
                        if audio_feature_ratio > 0:            #计算语音特征与图像特征的残差交叉注意力，同时给注意力乘以系数
                            # print('#'*5, norm_hidden_states.shape, audio_cond_fea.shape)
                            hidden_states = (
                                self.attn2(
                                    norm_hidden_states,
                                    encoder_hidden_states=audio_cond_fea, # B * 50 * 768，
                                    attention_mask=attention_mask,
                                ) * audio_feature_ratio         # 不懂？？ 为啥要乘以这个系数，是个超惨？
                                + hidden_states                 # 注意残差使用的是没有经过norm的图像特征
                            )
                        # print("Audio Cross-Attention max after:", hidden_states.max())


                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states      #进行mlp及残差

                    # Temporal-Attention
                    return hidden_states            #read 模式在这里返回
            # 如果是write模式，继续后面的代码, 注意write模式下没有交叉注意力，因为write的整个扩散网络的输入只有原始图，不需要做交叉注意力，只做自注意力就行
            if self.use_ada_layer_norm_zero:                        # 应该没进这里
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states             #进行一次残差，注意是没经过norm1的图像特征

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)          # 进行一次norm

            if self.use_ada_layer_norm_zero:                        #对特征进行缩放平移, 应该没进这里
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)                 # mlp网络计算

            if self.use_ada_layer_norm_zero:                        # 应该没进这里
                ff_output = gate_mlp.unsqueeze(1) * ff_output       # 特征缩放

            hidden_states = ff_output + hidden_states               # 算一个残差，注意hidden_states是前馈网络前没经过norm3的特征

            return hidden_states
        # ------------------------------这里开始不再是hacked_basic_transformer_inner_forward函数------------------------------------
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [        #遍历unet的mid和up的所有层，将其中的transformer block挑出来
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]            #按逆序进行排列
            )

            for i, module in enumerate(attn_modules):                                       #绑定新的forward推理函数
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(        #__get__函数的两个参数分别是实例和类本身
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, do_classifier_free_guidance=False, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [                 #将本read实例的TemporalBasicTransformerBlock层拿出来, 根据代码可知，update方法只会被 reader模式的实例调用
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [                 # 将传进来的writer实例的BasicTransformerBlock层拿出来
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(                                           # 对reader block进行排序
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(                                           # 对write block进行排序
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):              # 将read和write对应的每个atten层中，writer的bank列表中的所有特征复制给read的bank
                if do_classifier_free_guidance:
                    r.bank = [torch.cat([v, v]).to(dtype) for v in w.bank]
                else:
                    r.bank = [v.clone().to(dtype) for v in w.bank]

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
