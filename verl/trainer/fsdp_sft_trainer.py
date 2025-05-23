# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os
import random
import time
import numpy as np

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import boto3
import logging
import re
from contextlib import nullcontext
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto
from verl.utils.profiler import profile_training

from verl.third_party.trace_eval_new import evaluate_trace_response
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.sft_dataset import collate_fn
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj

# Taken from https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/data/data_helper.py#L93
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas
        logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(
                range(
                    (self.rank * self.num_samples), (self.rank + 1) * self.num_samples
                )
            )
        )[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter


class TrainerState(Stateful):
    """A wrapper for checkpointing the trainer state. This object is compliant with the Stateful protocol,
    so DCP will automatically call state_dict/load_state_dict as needed in the dcp.save/load APIs.
    """
    def __init__(self, model, optimizer, lr_scheduler, train_sampler, train_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.rng_state = None

    def state_dict(self):
        # Get model and optimizer state dicts
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        
        # Get lr scheduler state dict if exists
        if self.lr_scheduler is not None:
            lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        else:
            lr_scheduler_state_dict = None
            
        # Get sampler state dict
        train_sampler_state_dict = {
            'epoch': self.train_sampler.epoch,
            'start_iter': self.train_sampler.start_iter,
            'seed': self.train_sampler.seed
        }
        
        # Get dataloader state dict
        train_dataloader_state_dict = self.train_dataloader.state_dict()
        
        # Get RNG state
        rng_state = {
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }
        
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "lr_scheduler": lr_scheduler_state_dict,
            "train_sampler": train_sampler_state_dict,
            "train_dataloader": train_dataloader_state_dict,
            "rng": rng_state
        }

    def load_state_dict(self, state_dict):
        # Set model and optimizer state dicts
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"]
        )
        
        # Set lr scheduler state dict if exists
        if self.lr_scheduler is not None and state_dict["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            
        # Set sampler state dict
        if state_dict["train_sampler"] is not None:
            self.train_sampler.epoch = state_dict["train_sampler"]["epoch"]
            self.train_sampler.set_start_iter(state_dict["train_sampler"]["start_iter"])
            self.train_sampler.seed = state_dict["train_sampler"]["seed"]
            
        # Set dataloader state dict
        if state_dict["train_dataloader"] is not None:
            self.train_dataloader.load_state_dict(state_dict["train_dataloader"])
            
        # Set RNG state
        if state_dict["rng"] is not None:
            torch.set_rng_state(state_dict["rng"]['cpu'])
            torch.cuda.set_rng_state(state_dict["rng"]['cuda'])
            np.random.set_state(state_dict["rng"]['numpy'])
            random.setstate(state_dict["rng"]['random'])


class FSDPSFTTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # build tokenizer first
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        if self.config.data.chat_template is not None:
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self):
        config = self.config
        # build dataset
        from verl.utils.import_utils import load_extern_type

        # First check if a custom dataset class is specified
        if config.data.custom_cls.get("path", None):
            dataset_cls = load_extern_type(config.data.custom_cls.path, config.data.custom_cls.name)
        # Then check if multi-turn dataset should be used
        elif config.data.get('multiturn', {}).get('enable', False):
            dataset_cls = MultiTurnSFTDataset
        # Default to single-turn dataset
        else:
            dataset_cls = SFTDataset

        # Create datasets based on the selected class
        self.train_dataset = dataset_cls(parquet_files=config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         config=config.data)
        self.val_dataset = dataset_cls(parquet_files=config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       config=config.data)

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        self.train_sampler = StatefulDistributedSampler(dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            seed=config.trainer.seed if hasattr(config.trainer, 'seed') else 0,
                                                )
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=8,
                                           collate_fn=collate_fn,
                                           drop_last=True)

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                               config=config,
                                                                               torch_dtype=torch.float32,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, 'lr_scheduler') or self.config.optim.lr_scheduler == 'cosine':
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=num_warmup_steps,
                                                                num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == 'wsd':
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=num_warmup_steps,
                                                             num_training_steps=self.total_steps)
        else:
            raise ValueError(f'Unknown lr scheduler: {self.config.optim.lr_scheduler}')
        
        self.checkpoint_manager = FSDPCheckpointManager(model=self.fsdp_model,
                                                        optimizer=self.optimizer,
                                                        lr_scheduler=self.lr_scheduler,
                                                        tokenizer=self.tokenizer)

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        position_ids = batch['position_ids'].cuda()
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids[:, 1:].contiguous()
                    output = self.fsdp_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             position_ids=position_ids,
                                             use_cache=False)
                    logits = output.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels.contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss * loss_mask.to(loss.device)
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler

                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                               attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        use_cache=False)

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    # Gather and unpad for sequence parallelism
                    loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                    # This is the loss collected from all ulysses ranks
                    full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                                          indices=indices,
                                          batch=batch_size,
                                          seqlen=seqlen)
                    full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    full_loss = full_loss.reshape(-1)
                    loss_mask = loss_mask.to(full_loss.device)
                    loss = full_loss * loss_mask

                valid_token_this_rank = torch.sum(loss_mask)

                if self.config.data.balance_dp_token:
                    torch.distributed.all_reduce(valid_token_this_rank)
                    dp_size = self.ulysses_device_mesh.size('dp') if use_sp else torch.distributed.get_world_size()
                else:
                    dp_size = 1

                loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

                if do_backward:
                    loss.backward()
                return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage('Before optimizer step', logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}

    def validation_step(self, batch: TensorDict):
        """Calculates overall loss and evaluates plan/trace sections separately."""
        self.fsdp_model.eval()
        metrics = {}
        with torch.no_grad():
            # --- Input Preparation ---
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            position_ids = batch.get('position_ids', None)
            if position_ids is not None:
                position_ids = position_ids.cuda()

            labels = input_ids[:, 1:].contiguous()
            loss_mask = batch['loss_mask'][:, :-1].contiguous().cuda()

            # --- Forward Pass ---
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.fsdp_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         position_ids=position_ids,
                                         use_cache=False)
                logits = output.logits

            # --- Standard Loss Calculation ---
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            flat_shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            flat_shift_labels = shift_labels.view(-1)
            flat_loss_mask = loss_mask.view(-1)

            # Calculate per-token loss
            loss_unreduced = loss_fct(flat_shift_logits, flat_shift_labels.to(flat_shift_logits.device))
            overall_loss_masked = loss_unreduced * flat_loss_mask
            valid_overall_tokens = torch.sum(flat_loss_mask)

            # Handle token balancing if enabled
            dp_size = 1
            if self.config.data.balance_dp_token:
                dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
                torch.distributed.all_reduce(valid_overall_tokens)

            # Calculate average loss
            overall_loss = torch.sum(overall_loss_masked) / (valid_overall_tokens + 1e-8) * dp_size
            torch.distributed.all_reduce(overall_loss, op=torch.distributed.ReduceOp.AVG)
            """
            prompt= ['start', '1', '0', 'goal', '7', '3', 'wall', '7', '0', 'wall', '0', '1', 'wall', '2', '1', 'wall', '3', '1', 'wall', '8', '1', 'wall', '6', '2', 'wall', '0', '3', 'wall', '3', '3', 'wall', '4', '3', 'wall', '6', '3', 'wall', '0', '4', 'wall', '1', '4', 'wall', '2', '4', 'wall', '5', '4', 'wall', '0', '5', 'wall', '5', '5', 'wall', '8', '5', 'wall', '5', '6', 'wall', '6', '6', 'wall', '1', '7', 'wall', '2', '7', 'wall', '5', '7', 'wall', '7', '7', 'wall', '8', '7', 'wall', '4', '8', 'wall', '6', '8', 'wall', '7', '8', 'wall', '8', '8', 'wall', '0', '9', 'wall', '2', '9', 'wall', '4', '9', 'wall', '5', '9']
            response=['create', '1', '0', 'c0', 'c9', 'close', '1', '0', 'c0', 'c9', 'create', '2', '0', 'c1', 'c8', 'create', '0', '0', 'c1', 'c10', 'create', '1', '1', 'c1', 'c8', 'close', '2', '0', 'c1', 'c8', 'create', '3', '0', 'c2', 'c7', 'close', '1', '1', 'c1', 'c8', 'create', '1', '2', 'c2', 'c7', 'close', '1', '2', 'c2', 'c7', 'create', '1', '3', 'c3', 'c6', 'create', '2', '2', 'c3', 'c6', 'create', '0', '2', 'c3', 'c8', 'close', '2', '2', 'c3', 'c6', 'create', '2', '3', 'c4', 'c5', 'create', '3', '2', 'c4', 'c5', 'close', '3', '2', 'c4', 'c5', 'create', '4', '2', 'c5', 'c4', 'close', '4', '2', 'c5', 'c4', 'create', '5', '2', 'c6', 'c3', 'create', '4', '1', 'c6', 'c5', 'close', '3', '0', 'c2', 'c7', 'create', '4', '0', 'c3', 'c6', 'close', '5', '2', 'c6', 'c3', 'create', '5', '3', 'c7', 'c2', 'create', '5', '1', 'c7', 'c4', 'close', '2', '3', 'c4', 'c5', 'close', '1', '3', 'c3', 'c6', 'close', '5', '3', 'c7', 'c2', 'close', '4', '0', 'c3', 'c6', 'create', '5', '0', 'c4', 'c5', 'create', '4', '1', 'c4', 'c5', 'close', '5', '0', 'c4', 'c5', 'create', '6', '0', 'c5', 'c4', 'create', '5', '1', 'c5', 'c4', 'close', '4', '1', 'c4', 'c5', 'create', '5', '1', 'c5', 'c4', 'close', '5', '1', 'c5', 'c4', 'create', '6', '1', 'c6', 'c3', 'close', '6', '0', 'c5', 'c4', 'close', '6', '1', 'c6', 'c3', 'create', '7', '1', 'c7', 'c2', 'close', '5', '1', 'c5', 'c4', 'close', '7', '1', 'c7', 'c2', 'create', '7', '2', 'c8', 'c1', 'close', '7', '2', 'c8', 'c1', 'create', '8', '2', 'c9', 'c2', 'create', '7', '3', 'c9', 'c0', 'close', '7', '3', 'c9', 'c0', 'plan', '1', '0', 'plan', '2', '0', 'plan', '3', '0', 'plan', '4', '0', 'plan', '4', '1', 'plan', '5', '1', 'plan', '6', '1', 'plan', '7', '1', 'plan', '7', '2', 'plan', '7', '3']
            labels = prompt+response
            """
            # --- Plan/Trace Evaluation ---
            # Decode model outputs and labels to text
            outputs_text = self.tokenizer.batch_decode(output.logits.argmax(dim=-1), skip_special_tokens=True)
            labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Get prompt and response from labels_text 
            
                    

            # Initialize evaluation metrics
            plan_metrics = {'accuracy': 0.0}
            trace_metrics = {'accuracy': 0.0}
            total_samples = len(outputs_text)

            # Evaluate each sample
            for output_text, label_text in zip(outputs_text, labels_text):
                prompt = []
                p_start = False
                p_end = False
                for i in range(len(label_text)):
                    if label_text[i] == 'start':
                        p_start = True
                    if 'Assistant' in label_text[i]:
                        p_end = True
                    if p_start and not p_end:
                        prompt.append(label_text[i])
                    
                #Convert list to string
                output = ''.join(output_text)
                prompt = ''.join(prompt)
                
                try:
                    trace_is_valid, llm_plan_is_valid, errors, llm_plan_errors = evaluate_trace_response([i.strip() for i in output.split(' ')], [i.strip() for i in prompt.split(' ')], [self.config.trainer.maze_size, self.config.trainer.maze_size], None, False)
                except Exception as e:
                    print(prompt)
                    print("\n\nEND OF PROMPT", type(prompt), '\n\n')
                    print(output)
                    print("\n\nEND OF OUTPUT", type(output), '\n\n')
                    trace_is_valid = False
                    llm_plan_is_valid = False

                if trace_is_valid:
                    trace_metrics['accuracy'] += 1
                if llm_plan_is_valid:
                    plan_metrics['accuracy'] += 1

            # Average the metrics
            if total_samples > 0:
                plan_metrics['accuracy'] /= total_samples
                trace_metrics['accuracy'] /= total_samples

            # Combine all metrics
            metrics = {
                'val/loss': overall_loss.item(),
                'val/plan_accuracy': plan_metrics['accuracy'],
                'val/trace_accuracy': trace_metrics['accuracy']
            }

        return metrics
    
    @staticmethod
    def get_rng_state():
        rng_state = {
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state['cpu'])
        torch.cuda.set_rng_state(rng_state['cuda'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['random'])
    def save_checkpoint(self, step):
        """Save model checkpoint using Distributed Checkpoint (DCP)."""
        try:
            path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
            
            # Create checkpoint directory
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
            
            # Create trainer state
            trainer_state = TrainerState(
                model=self.fsdp_model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_sampler=self.train_sampler,
                train_dataloader=self.train_dataloader
            )
            
            # Prepare state dict for DCP
            state_dict = {
                "trainer": trainer_state
            }
            
            # Save using DCP
            dcp.save(
                state_dict=state_dict,
                checkpoint_id=path,
            )
            
            # Update latest checkpoint iteration tracker
            if self.device_mesh.get_rank() == 0:
                local_latest_checkpointed_iteration = os.path.join(
                    self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
                )
                with open(local_latest_checkpointed_iteration, "w") as f:
                    f.write(str(step))
            
            # Upload to HDFS if configured
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
            
            # Ensure all ranks wait for checkpoint to complete
            torch.distributed.barrier()
            
            # Sync to S3 if configured
            if self.config.trainer.s3_checkpoint_dir:
                result = os.system(f"aws s3 sync {self.config.trainer.default_local_dir} {self.config.trainer.s3_checkpoint_dir}")
                if result == 0:
                    logger.info(f"Checkpoint sync to S3 initiated for step {step}")
                else:
                    logger.error(f"S3 sync command failed with exit code {result} at step {step}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint at step {step}: {str(e)}")
            raise

    def _load_checkpoint(self):
        """Load model checkpoint using Distributed Checkpoint (DCP)."""
        if self.config.trainer.resume_mode == 'disable':
            return 0

        try:
            # Find checkpoint directory
            if self.config.trainer.default_local_dir is not None:
                checkpoint_folder = self.config.trainer.default_local_dir
                if not os.path.isabs(checkpoint_folder):
                    working_dir = os.getcwd()
                    checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)

            # Determine which checkpoint to load
            if self.config.trainer.resume_mode == 'auto':
                if global_step_folder is None:
                    logger.info('Training from scratch')
                    return 0
            else:
                if self.config.trainer.resume_mode == "resume_path":
                    assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                    assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                    global_step_folder = self.config.trainer.resume_from_path
                    if not os.path.isabs(global_step_folder):
                        working_dir = os.getcwd()
                        global_step_folder = os.path.join(working_dir, global_step_folder)
                    logger.info(f'Load from checkpoint folder: {global_step_folder}')
            
            # Validate checkpoint directory exists
            if not os.path.exists(global_step_folder):
                raise FileNotFoundError(f"Checkpoint directory not found: {global_step_folder}")
            
            # Create trainer state
            trainer_state = TrainerState(
                model=self.fsdp_model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_sampler=self.train_sampler,
                train_dataloader=self.train_dataloader
            )
            
            # Prepare state dict for loading
            state_dict = {
                "trainer": trainer_state
            }
            
            # Load using DCP
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=global_step_folder,
            )
            
            # Set global step
            global_step = int(global_step_folder.split('global_step_')[-1])
            logger.info(f'Setting global step to {global_step}')
            logger.info(f'Resuming from {global_step_folder}')
            
            return global_step
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    @profile_training
    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        # Load checkpoint before starting training
        global_step = self._load_checkpoint()

        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(self.train_dataloader,
                             total=self.steps_per_epoch,
                             desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}"):
                global_step += 1
                # Update profiler step
                if hasattr(self, '_profiler'):
                    self._profiler.update_step(global_step)
                
                # Start timing the step
                step_start_time = time.time()
                
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                
                # Calculate step duration
                step_duration = time.time() - step_start_time
                metric['train/step_duration'] = step_duration
                
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                # Save checkpoint at regular intervals if save_freq is set
                if hasattr(self.config.trainer, 'save_freq') and self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0:
                    self.save_checkpoint(step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    val_metrics_list = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                        val_metrics = self.validation_step(val_data)
                        val_metrics_list.append(val_metrics)
                    if rank == 0:
                        # Average metrics across validation batches
                        avg_val_metrics = {}
                        for key in val_metrics_list[0].keys():
                            avg_val_metrics[key] = torch.mean(torch.tensor([m[key] for m in val_metrics_list])).item()
                        tracking.log(data=avg_val_metrics, step=global_step)
                    torch.distributed.barrier()

                    # Save final checkpoint
                    self.save_checkpoint(step=global_step)
                    return

            # validation
            val_metrics_list = []
            for data in self.val_dataloader:
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                val_metrics = self.validation_step(data)
                val_metrics_list.append(val_metrics)
            if rank == 0:
                # Average metrics across validation batches
                avg_val_metrics = {}
                if val_metrics_list:
                    for key in val_metrics_list[0].keys():
                        avg_val_metrics[key] = torch.mean(torch.tensor([m[key] for m in val_metrics_list])).item()
                    tracking.log(data=avg_val_metrics, step=global_step)
                else:
                    logger.warning("Validation dataloader was empty, skipping validation logging.")
            torch.distributed.barrier()

            # save checkpoint at end of epoch if save_freq is not set or if it's not a multiple of save_freq
            if not hasattr(self.config.trainer, 'save_freq') or self.config.trainer.save_freq <= 0 or global_step % self.config.trainer.save_freq != 0:
                self.save_checkpoint(step=global_step)


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
