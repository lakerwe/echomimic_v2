import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm

from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method


@dataclass
class EchoMimicV2PipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class EchoMimicV2Pipeline(DiffusionPipeline):

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        audio_guider,
        pose_encoder,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,                  # UNet2DConditionModel
            denoising_unet=denoising_unet,                  # EMOUNet3DConditionModel
            audio_guider=audio_guider,                      # whisper 模型
            pose_encoder=pose_encoder,                      # 里面是一系列2d卷积层，将手势图卷出来一个embedding
            scheduler=scheduler,                            # 扩散采样器
            image_proj_model=image_proj_model,              #None
            tokenizer=tokenizer,                            #None
            text_encoder=text_encoder,                      #None
            # audio_feature_mapper=audio_feature_mapper
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.ref_image_processor = VaeImageProcessor(       # vae 的预处理模块
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents_bp(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        context_frame_length
    ):
        shape = (
            batch_size,
            num_channels_latents,
            # context_frame_length,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents_seg = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )
        latents = latents_seg
        
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        print(f"latents shape:{latents.shape}, video_length:{video_length}")
        return latents
    def prepare_latents_smooth(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        context_frame_length
    ):
        shape = (
            batch_size,
            num_channels_latents,
            # context_frame_length,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents_seg = randn_tensor(                                     # generator是随机种子，即torch.manual_seed()返回的生成器?控制噪声的生成
            shape, generator=generator, device=device, dtype=dtype
        )

        latents = latents_seg
        
        latents = torch.clamp(latents_seg, -1.5, 1.5)


        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        print(f"latents shape:{latents.shape}, video_length:{video_length}")
        
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image,                                                                  #输入的图片
        audio_path,                                                                 # 音频地址
        poses_tensor,                                                               # 手部动作张量
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,                                                             # 2.5
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=0,
        context_batch_size=1,
        interpolation_factor=1,
        audio_sample_rate=16000,
        fps=25,
        audio_margin=2,
        start_idx=0,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0                          # 这里是为什么？不懂??   是否进行分类引导

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)            # 设置推理时间步长
        timesteps = self.scheduler.timesteps
        # print("timesteps :",timesteps)  #timesteps为 tensor([999, 966, 932, 899, 866, 832, 799, 766, 732, 699, 666, 632, 599, 566,532, 499, 466, 432, 399, 366, 332, 299, 266, 232, 199, 166, 132,  99, 66,  32], device='cuda:0')

        batch_size = 1

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,                                                    # UNet2DConditionModel类,只用来write
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",                                                           # 不懂
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,                                                    # EMOUNET3DConditionModel类,只用来read
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        # 处理音频，利用whisper进行音频编码，并切分,转换为张量
        whisper_feature = self.audio_guider.audio2feat(audio_path)

        whisper_chunks = self.audio_guider.feature2chunks(feature_array=whisper_feature, fps=fps)
        audio_frame_num = whisper_chunks.shape[0]
        audio_fea_final = torch.Tensor(whisper_chunks).to(dtype=self.vae.dtype, device=self.vae.device)
        audio_fea_final = audio_fea_final.unsqueeze(0)
        
        video_length = min(video_length, audio_frame_num)                           #确定视频长度
        
        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents_smooth(                                  #生成一个随机噪声张量,张量latents.shape为 torch.Size([1, 4, 149, 96, 96])
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            audio_fea_final.dtype,
            device,
            generator,
            context_frames
        )
        # 对手式图片进行编码,       注意pose_encoder是一系列2d卷积层，难道是自己训练的一个网络??? 不懂
        pose_enocder_tensor = self.pose_encoder(poses_tensor)                   # 这里poses_tensor.shape为[1, 3, 149, 768, 768],pose_enocder_tensor.shape为 [1, 320, 149, 96, 96]
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)      #这里判断采样器中是否有这两个参数，如果有，将这两个参数的值赋值进去

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(                 # 用vae的预处理模块对原始图片进行预处理
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(                                 #将预处理的图片放到cuda中
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean  # 注意这里将图像编码后潜空间的分布均值拿出来 , 注意ref_image_latents.shape为[1, 4, 96, 96]
        ref_image_latents = ref_image_latents * 0.18215  # (b , 4, h, w)        # 为啥要在这里乘以一个系数?? 不懂？？
        context_scheduler = get_context_scheduler(context_schedule)             # context_schedule 为 'uniform', get_context_scheduler函数返回context.py的uniform函数

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order          #计算一个热启动迭代步数,self.scheduler.order是在diffusers库的类中自身定义的,值为1
        context_queue = list(                                                   # 本例产生的是连续12帧
            context_scheduler(                                                  # 这里怎么做的？不懂？？
                0,
                num_inference_steps,                                            # 这里是 30
                latents.shape[2],                                               # 要生成的帧数,149
                context_frames,
                context_stride,
                context_overlap,
            )
        )
        # context_queue输出为
        # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], 
        #  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56], [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], 
        #  [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74], [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83], [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92], 
        #  [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101], [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119], 
        #  [117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128], [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137], [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146], 
        #  [144, 145, 146, 147, 148, 0, 1, 2, 3, 4, 5, 6]]
# 这里 timesteps 为 tensor([999,966,932,899, 866, 832, 799, 766, 732, 699, 666, 632, 599, 566, 532, 499, 466, 432, 399, 366, 332, 299, 266, 232, 199, 166, 132,  99,66,  32], device='cuda:0')
# timesteps长度为 30
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t_i, t in enumerate(timesteps):

                noise_pred = torch.zeros(                                           #用来记录单次推理预测的每一帧的噪声
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(                                              #用来记录每一帧推理的次数，本实验默认是30次
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image                            #第一次扩散推理时，将原始图片的vae编码张量送到UNet2dconditionalModel进行一次推理，注意没有输入其他任何条件(唯一条件就是时间0)，同时将所有的atten层保存在bank中
                if t_i == 0:                                            #只在第一次对原始图片利用UNet2dconditionalModel进行一次编码,保存在write 的bank中
                    self.reference_unet(                                # 注意这里已经把self.reference_unet中所有的attention层改了，attention走的hacked_basic_transformer_inner_forward
                        ref_image_latents,                              #送入vae编码的图片
                        torch.zeros_like(t),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer, do_classifier_free_guidance=True)     #将write的bank更新到read的bank中

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)        #本次测试len(context_queue)=17,context_batch_size 为 1,所以num_context_batches为17

                global_context = []
                for j in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            j * context_batch_size : (j + 1) * context_batch_size
                        ]
                    )
                # 第一次迭代 global_context 的值为, 相当于给context_queue的每个原始加了一层列表
                # [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], [[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], [[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]], [[27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]], 
                #  [[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]], [[45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]], [[54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]], 
                # [[63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]], [[72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]], [[81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]], 
                # [[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]], [[99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]], [[108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]], 
                # [[117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]], [[126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137]], 
                # [[135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]], [[144, 145, 146, 147, 148, 0, 1, 2, 3, 4, 5, 6]]]

                ## refine
                for context in global_context:
                    new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
                    for c_j in range(len(context)):
                        for c_i in range(len(context[c_j])):
                            new_context[c_j][c_i] = (context[c_j][c_i] + t_i * 3) % video_length
        

                    latent_model_input = (                                  # 将latents的某些帧拿出来cat到一起
                        torch.cat([latents[:, :, c] for c in new_context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )

                    audio_latents_cond = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)         #将对应帧的audio_feature拿出来cat到一起
                                        
                    audio_latents = torch.cat([torch.zeros_like(audio_latents_cond), audio_latents_cond], 0)        # 将全0的张量与音频特征cat到一起
                    pose_latents_cond = torch.cat([pose_enocder_tensor[:, :, c] for c in new_context]).to(device)   # 将对应帧的pose feature拿出来cat到一起
                    pose_latents = torch.cat([torch.zeros_like(pose_latents_cond), pose_latents_cond], 0)           # 将全0的张量与pose feature cat到一起
                    la = latent_model_input
                    latent_model_input = self.scheduler.scale_model_input(                                          # 将噪声特征与当前步数输入 scheduler
                        latent_model_input, t                                                                       # 查看diffusers库的scale_model_input函数，发现该函数什么都没做，直接将输入的latent_model_input返回了
                    )
                    b, c, f, h, w = latent_model_input.shape
                    
                    pred = self.denoising_unet(                                                                     #过一遍emo的unet
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        audio_cond_fea=audio_latents if do_classifier_free_guidance else audio_latents_cond,
                        face_musk_fea=pose_latents if do_classifier_free_guidance else pose_latents_cond,
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(new_context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred                                            #记录每次预测的噪声
                        counter[:, :, c] = counter[:, :, c] + 1                                                     #记录每帧预测的次数

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(                                                                      # 去噪过程
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if t_i == len(timesteps) - 1 or (
                    (t_i + 1) > num_warmup_steps and (t_i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

            reference_control_reader.clear()
            reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)
        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return EchoMimicV2PipelineOutput(videos=images)
