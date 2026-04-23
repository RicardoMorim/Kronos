import os
os.environ["USE_LIBUV"] = "0"
import sys
import json
import time
import math
from time import gmtime, strftime
from pathlib import Path
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import comet_ml

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    is_distributed,
    set_seed,
    get_model_size,
    format_time
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    batch_size = int(config.get('predictor_batch_size', config['batch_size']))
    num_workers = int(config.get('predictor_num_workers', config.get('num_workers', config.get('workers', 0))))
    pin_memory = bool(config.get('pin_memory', True))
    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': True,
    }
    val_dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': False,
    }
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        val_dataloader_kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        prefetch_factor = config.get('prefetch_factor', None)
        if prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = int(prefetch_factor)
            val_dataloader_kwargs['prefetch_factor'] = int(prefetch_factor)

    use_ddp = world_size > 1 and is_distributed()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), **dataloader_kwargs
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=val_sampler,
        **val_dataloader_kwargs
    )

    if rank == 0:
        print(
            f"Predictor DataLoader config -> batch_size={batch_size}, num_workers={num_workers}, "
            f"pin_memory={pin_memory}, persistent_workers={dataloader_kwargs.get('persistent_workers', False)}"
        )

    return train_loader, val_loader, train_dataset, valid_dataset


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def train_model(model, tokenizer, device, config, save_dir, logger, rank, world_size):
    start_time = time.time()
    predictor_batch_size = int(config.get('predictor_batch_size', config['batch_size']))
    accumulation_steps = max(1, int(config.get('predictor_accumulation_steps', 1)))
    use_amp = bool(config.get('use_amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    tokenizer_device = next(tokenizer.parameters()).device

    if rank == 0:
        effective_bs = predictor_batch_size * world_size * accumulation_steps
        print(f"Effective Predictor BATCHSIZE per GPU: {predictor_batch_size}, Total effective: {effective_bs}")
        print(f"Predictor AMP enabled: {use_amp}, accumulation_steps: {accumulation_steps}")
        print(f"Tokenizer device for predictor training: {tokenizer_device}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=max(1, math.ceil(len(train_loader) / accumulation_steps)),
        epochs=config['epochs'],
        pct_start=0.1,
        div_factor=10
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                batch_x_for_tokenizer = batch_x.to(tokenizer_device, non_blocking=(tokenizer_device.type == 'cuda'))
                if tokenizer_device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        token_seq_0, token_seq_1 = tokenizer.encode(batch_x_for_tokenizer, half=True)
                else:
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x_for_tokenizer, half=True)

            token_seq_0 = token_seq_0.to(device, non_blocking=True)
            token_seq_1 = token_seq_1.to(device, non_blocking=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = unwrap_model(model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, s1_loss, s2_loss = unwrap_model(model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            loss_scaled = loss / accumulation_steps
            scaler.scale(loss_scaled).backward()
            should_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if rank == 0 and logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                batch_x_for_tokenizer = batch_x.to(tokenizer_device, non_blocking=(tokenizer_device.type == 'cuda'))
                if tokenizer_device.type == 'cuda':
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        token_seq_0, token_seq_1 = tokenizer.encode(batch_x_for_tokenizer, half=True)
                else:
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x_for_tokenizer, half=True)

                token_seq_0 = token_seq_0.to(device, non_blocking=True)
                token_seq_1 = token_seq_1.to(device, non_blocking=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = unwrap_model(model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    val_loss, _, _ = unwrap_model(model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
        val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
        if is_distributed():
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_sum_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0

        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                unwrap_model(model).save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

            if device.type == 'cuda' and bool(config.get('empty_cuda_cache_each_epoch', True)):
                torch.cuda.empty_cache()

        if is_distributed():
            dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    return dt_result


def main(config: dict):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])

    comet_logger, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        comet_api_key = config.get('comet_config', {}).get('api_key')
        comet_workspace = config.get('comet_config', {}).get('workspace')
        if config['use_comet'] and comet_api_key and comet_workspace:
            comet_logger = comet_ml.Experiment(
                api_key=comet_api_key,
                project_name=config['comet_config']['project_name'],
                workspace=comet_workspace,
            )
            comet_logger.add_tag(config['comet_tag'])
            comet_logger.set_name(config['comet_name'])
            comet_logger.log_parameters(config)
            print("Comet Logger Initialized.")
        elif rank == 0 and config['use_comet']:
            print("Comet logging requested but API key/workspace missing; continuing without Comet.")

    if is_distributed():
        dist.barrier()

    tokenizer_path = config['finetuned_tokenizer_path']
    if not Path(tokenizer_path).exists():
        print(f"Tokenizer checkpoint not found at {tokenizer_path}; falling back to {config['pretrained_tokenizer_path']}.")
        tokenizer_path = config['pretrained_tokenizer_path']

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer_device_cfg = str(config.get('predictor_tokenizer_device', 'cpu')).strip().lower()
    if tokenizer_device_cfg.startswith('cuda') and torch.cuda.is_available():
        tokenizer_device = device
    else:
        tokenizer_device = torch.device('cpu')
    tokenizer.eval().to(tokenizer_device)
    tokenizer.requires_grad_(False)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)
    if world_size > 1 and is_distributed():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Predictor Model Size: {get_model_size(unwrap_model(model))}")
        print(f"Predictor model device: {device}, tokenizer device: {tokenizer_device}")

    dt_result = train_model(
        model, tokenizer, device, config, save_dir, comet_logger, rank, world_size
    )

    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if comet_logger: comet_logger.end()

    cleanup_ddp()


if __name__ == '__main__':
    if "KRONOS_CONFIG_PROFILE" not in os.environ:
        os.environ["KRONOS_CONFIG_PROFILE"] = "bybit"

    config_instance = Config()
    main(config_instance.__dict__)
