import os
import math
import time
import datetime
import logging
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from utils.ddp import distributed_setup, info_if_rank_zero

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig
import wandb
from tqdm import tqdm 
from dataset.vos_dataset_0709_2010 import TenCamusDataset
from model.trainer import Trainer
from utils.logger import TensorboardLogger


log = logging.getLogger(__name__)


@hydra.main(version_base='1.3.2', config_path='configs', config_name='echonet_full')
def train(cfg: DictConfig):
    os.environ["WANDB_MODE"] = "offline"

    # local_rank, world_size = distributed_setup()  # auto from torchrun/DDP env
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)


    if local_rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        cfg_dict_wandb = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="EchoMAP",
            name=f"run_{timestamp}", 
            config=cfg_dict_wandb,  # log all cfg params
            reinit=True  # allow multiple wandb.init() calls
        )
        wandb.config.update(cfg_dict_wandb, allow_val_change=True)  # ensure cfg recorded
    
    try:
        run_dir = HydraConfig.get().run.dir
        if local_rank == 0:
            info_if_rank_zero(f'All configuration: {cfg}')
            info_if_rank_zero(f'Number of detected GPUs: {world_size}')
            info_if_rank_zero(f'Run dir: {run_dir}')

        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True


        if local_rank == 0:
            info_if_rank_zero(f'batch_size={cfg.main_training.batch_size}')
        cfg.main_training.batch_size = max(cfg.main_training.batch_size // world_size, 1)
        if local_rank == 0:
            info_if_rank_zero(f'batch_size(per-GPU)={cfg.main_training.batch_size}')

        if local_rank == 0:
            info_if_rank_zero(f'num_workers={cfg.main_training.num_workers}')
        cfg.main_training.num_workers = max(cfg.main_training.num_workers // world_size, 1)
        if local_rank == 0:
            info_if_rank_zero(f'num_workers(per-GPU)={cfg.main_training.num_workers}')

        # logger (optional)
        log_writer = TensorboardLogger(run_dir, logging.getLogger(), enabled_tb=(local_rank == 0))

        # ===== build dataloaders =====
        def create_safe_dataloader(dataset, sampler, batch_size, num_workers, pin_memory=False, drop_last=False):
            """DataLoader with auto-fallback when multi-process fails"""
            try:
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=False,
                    drop_last=drop_last
                )
            except Exception as e:
                if local_rank == 0:
                    log.warning(f"Failed to create DataLoader with num_workers={num_workers}, trying with num_workers=0: {e}")
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=0,  # fallback to single-process
                    pin_memory=False,
                    shuffle=False,
                    drop_last=drop_last
                )

        # train split
        train_dataset = TenCamusDataset(
            filepath=cfg.data_path,
            mode='train',
            seq_length=cfg.main_training.seq_length,
            max_num_obj=cfg.main_training.num_objects,
            size=cfg.main_training.crop_size[0],
            sparse_gt=cfg.get('data_sparse_gt', False)
        )
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        train_loader = create_safe_dataloader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=cfg.main_training.batch_size,
            num_workers=cfg.main_training.num_workers,
            pin_memory=False,
            drop_last=True
        )

        # val split
        val_dataset = TenCamusDataset(
            filepath=cfg.data_path,
            mode='val',
            seq_length=cfg.main_training.seq_length,
            max_num_obj=cfg.main_training.num_objects,
            size=cfg.main_training.crop_size[0],
            sparse_gt=cfg.get('data_sparse_gt', False)
        )
        val_sampler = data.distributed.DistributedSampler(val_dataset)
        val_loader = create_safe_dataloader(
            dataset=val_dataset,
            sampler=val_sampler,
            batch_size=cfg.main_training.batch_size,
            num_workers=cfg.main_training.num_workers,
            pin_memory=False,
            drop_last=False
        )

        test_dataset = TenCamusDataset(
            filepath=cfg.data_path,
            mode='test',
            seq_length=cfg.main_training.seq_length,
            max_num_obj=cfg.main_training.num_objects,
            size=cfg.main_training.crop_size[0],
            sparse_gt=cfg.get('data_sparse_gt', False)
        )
        test_sampler = data.distributed.DistributedSampler(test_dataset)
        test_loader = create_safe_dataloader(
            dataset=test_dataset,
            sampler=test_sampler,
            batch_size=cfg.main_training.batch_size,
            num_workers=cfg.main_training.num_workers,
            pin_memory=False,
            drop_last=False
        )

        # ===== UDA Stage 1 target loader (guarded: only when cfg.uda.enable) =====
        target_loader = None
        uda_cfg = cfg.get('uda', None)
        if uda_cfg is not None and uda_cfg.get('enable', False):
            t_bs = max(int(uda_cfg.get('target_batch_size', 2)) // world_size, 1)
            t_workers = max(int(uda_cfg.get('target_num_workers',
                                            cfg.main_training.num_workers)) // world_size, 1)
            # leakage guard: target train split must not overlap val/test
            import json as _json
            with open(os.path.join(uda_cfg.target_data_path,
                                   'camus_public_datasplit_20250706.json')) as _f:
                _sp = _json.load(_f)
            _tr = set(_sp['train_data'])
            assert not (_tr & set(_sp['val_data'])), 'UDA target train overlaps val!'
            assert not (_tr & set(_sp['test_data'])), 'UDA target train overlaps test!'
            if local_rank == 0:
                info_if_rank_zero(
                    f"[UDA] target={uda_cfg.target_data_path} "
                    f"train={len(_tr)} (disjoint from val/test), "
                    f"batch={t_bs}, hide_gt={uda_cfg.get('target_hide_gt', True)}")
            target_dataset = TenCamusDataset(
                filepath=uda_cfg.target_data_path,
                mode='train',
                seq_length=int(uda_cfg.get('target_seq_length',
                                           cfg.main_training.seq_length)),
                max_num_obj=cfg.main_training.num_objects,
                size=cfg.main_training.crop_size[0],
                sparse_gt=bool(uda_cfg.get('target_sparse_gt', True)),
                hide_gt=bool(uda_cfg.get('target_hide_gt', True))
            )
            target_sampler = DistributedSampler(target_dataset)
            # dedicated RNG generator: target-side iterator creation must not
            # shift the global torch RNG stream (keeps lambda_self=0 runs
            # bit-comparable to source-only runs)
            tgt_gen = torch.Generator()
            tgt_gen.manual_seed(cfg.seed + 12345)
            target_loader = data.DataLoader(
                dataset=target_dataset,
                sampler=target_sampler,
                batch_size=t_bs,
                num_workers=t_workers,
                pin_memory=False,
                shuffle=False,
                drop_last=True,
                generator=tgt_gen,
            )
            # leakage test: one real batch must have all-zero GT tensors.
            # num_workers=0 runs __getitem__ in the MAIN process (python
            # random / torch RNG) -- snapshot & restore RNG state so the
            # global training stream stays identical to source-only runs.
            if local_rank == 0:
                _rng_t = torch.get_rng_state()
                _rng_py = random.getstate()
                _rng_np = np.random.get_state()
                _chk = data.DataLoader(target_dataset, batch_size=t_bs,
                                       shuffle=True, num_workers=0,
                                       generator=torch.Generator().manual_seed(cfg.seed + 999))
                _b = next(iter(_chk))
                assert float(_b['ff_gt'].abs().sum()) == 0.0, 'ff_gt leaked!'
                assert float(_b['cls_gt'].abs().sum()) == 0.0, 'cls_gt leaked!'
                assert float(_b['gt_valid'].abs().sum()) == 0.0, 'gt_valid leaked!'
                torch.set_rng_state(_rng_t)
                random.setstate(_rng_py)
                np.random.set_state(_rng_np)
                info_if_rank_zero('[UDA] leakage check OK: target batch '
                                  'ff_gt/cls_gt/gt_valid all zero')

        # seeding happens AFTER all dataset/leak-check work: the UDA leakage
        # check runs __getitem__ in the main process, so it must consume RNG
        # before seeding to keep the training stream identical to source-only
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # ===== build Trainer =====
        stage_cfg = cfg.main_training
        trainer = Trainer(
            cfg=cfg,
            stage_cfg=stage_cfg,
            log=log_writer,
            run_path=run_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            target_loader=target_loader
        )

        total_iterations = stage_cfg.num_iterations
        it = 0
        current_epoch = 0
        max_epoch = math.ceil(total_iterations / len(train_loader))
        if local_rank == 0:
            log.info(f"Total iterations: {total_iterations}")
            log.info(f"train_loader length (batches per epoch): {len(train_loader)}")
            info_if_rank_zero(f"Total iteration={total_iterations}, est. epochs={max_epoch} ...")

        # ===== tqdm progress bar (only on local_rank=0) =====
        if local_rank == 0:
            pbar = tqdm(total=total_iterations, desc="Training", ncols=120)
        else:
            pbar = None  # no progress bar on non-main ranks


        # iteration-based training
        steps_per_epoch = len(train_loader)
        data_iter = iter(train_loader)

        for it in range(total_iterations):
            # reset data_iter at epoch boundary
            current_epoch = it // steps_per_epoch + 1

            try:
                batch_data = next(data_iter)
            except StopIteration:
                train_sampler.set_epoch(current_epoch)
                data_iter = iter(train_loader)
                batch_data = next(data_iter)


            loss_val = trainer.do_pass(batch_data, it)

            # ===== save checkpoint =====
            if cfg.save == 1:
                if local_rank == 0 :
                    if it % cfg.save_weights_interval == 0 and it > 0:
                        trainer.save_weights(it)
                        log.info(f"Weights saved at iteration {it} and epoch {current_epoch}")
                    if it % cfg.save_checkpoint_interval == 0 and it > 0:
                        trainer.save_checkpoint(it)
                        log.info(f"Checkpoint saved at iteration {it} and epoch {current_epoch}")

            # update tqdm
            if local_rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'iter': it + 1,
                    'loss': f"{loss_val:.4f}"
                })

            # ===== periodic evaluation =====
            if (it + 1) % cfg.eval_stage.eval_interval == 0:
                trainer.evaluate(
                    val_loader=val_loader,
                    epoch=current_epoch,    # epoch param is consumed by eval_vos / sampler; rename if needed
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1
                )

                trainer.test(
                    test_loader=test_loader,
                    epoch=current_epoch,    # epoch param is consumed by eval_vos / sampler; rename if needed
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1
                )

        info_if_rank_zero("Training completed. ")

        # final weights save (the % interval never fires on the last iteration)
        if cfg.save == 1 and local_rank == 0:
            trainer.save_weights(total_iterations - 1)
            log.info(f"Final weights saved at iteration {total_iterations - 1}")

        

    finally:
        if local_rank == 0:
            wandb.finish()
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
