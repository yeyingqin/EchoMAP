import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np

import wandb
from omegaconf import DictConfig
from utils.logger import TensorboardLogger
from utils.log_integrator import Integrator
from utils.time_estimator import TimeEstimator
import PIL

from model.backbone import Backbone
from model.utils.parameter_groups import get_parameter_groups
from model.losses import LossComputer
from model.uda.cfmc_teacher import CFMCTeacher, frame_entropy
from model.uda.uda_losses import pseudo_label_loss
from model.uda.temporal_consistency import area_smooth_loss
from model.uda.prototype_alignment import prototype_alignment_loss
from model.uda.contrast_proto import (contrast_proto_alignment_loss,
                                      contrast_transform)
from metr.dice import dice_coefficient, sespiou_coefficient
from vis.vis0709_1020 import visualize_sequence

class Trainer:
    def __init__(self,
                 cfg: DictConfig,
                 stage_cfg: DictConfig,
                 log: TensorboardLogger,
                 run_path: str,
                 train_loader,
                 val_loader,
                 test_loader,
                 target_loader=None):
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.log = log
        self.run_path = run_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.exp_id = cfg['exp_id']
        self.stage = stage_cfg['name']
        self.use_amp = stage_cfg.amp

        local_rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        # build model; optionally initialize from a checkpoint (finetuning)
        model = Backbone(
            use_memory_cache=cfg.model.get('use_memory_cache', False),
            cache_interval=cfg.model.get('cache_interval', 10),
            detach_memory_cache=cfg.model.get('detach_memory_cache', True),
            memory_cache_context=cfg.model.get('memory_cache_context', 'gap_key'),
            kpff_enhancer=(cfg.model.get('kpff_enhancer', None)),
            kpff_wavelet=(cfg.model.get('kpff_wavelet', None)),
            kpff_joint=(cfg.model.get('kpff_joint', None)),
            kpff_context=(cfg.model.get('kpff_context', None)),
        ).cuda()
        init_weights = stage_cfg.get('init_weights', '')
        if init_weights:
            sd = torch.load(init_weights, map_location=f'cuda:{local_rank}')
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if local_rank == 0:
                print(f'[init_weights] loaded from {init_weights} '
                      f'(missing={len(missing)}, unexpected={len(unexpected)})')
                # when a newly added module is enabled, only ITS parameters
                # may be missing from the old checkpoint -- surface anything
                # else instead of hiding it behind the count
                allowed_prefixes = []
                if model.KPFF.wavelet_gate is not None:
                    allowed_prefixes.append('KPFF.wavelet_gate.')
                if model.KPFF.p_gate is not None:
                    allowed_prefixes.append('KPFF.p_gate.')
                if allowed_prefixes:
                    stray = [k for k in missing
                             if not any(k.startswith(a) for a in allowed_prefixes)]
                    if stray:
                        print(f'[init_weights][warn] missing keys outside '
                              f'{allowed_prefixes}: {stray}')

        # optionally freeze everything except the memory cache routing projections
        self.freeze_memory_cache_only = bool(
            stage_cfg.get('freeze_except_memory_cache', False))
        if self.freeze_memory_cache_only:
            allowed = ('memory_cache_ctx_proj', 'memory_cache_q_proj',
                       'memory_cache_k_proj')
            for name, param in model.named_parameters():
                param.requires_grad = any(name == a or name.startswith(a + '.')
                                          for a in allowed)
            trainable = [(n, p.numel()) for n, p in model.named_parameters()
                         if p.requires_grad]
            n_train = sum(n for _, n in trainable)
            assert n_train > 0, 'freeze_except_memory_cache: no trainable params found'
            print(f'[freeze_except_memory_cache] trainable params: {n_train} '
                  f'in {len(trainable)} tensors:')
            for n, cnt in trainable:
                print(f'  {n}  ({cnt})')
            self._freeze_bn_eval(model)   # keep BN running stats frozen

        self.model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            # sparse-GT batches may leave some windows with only frame-0
            # supervision (ES outside the window); allow unused-parameter
            # iterations instead of crashing the reducer
            find_unused_parameters=True
        )
        self.size = stage_cfg['crop_size']

        if local_rank == 0:
            # log="parameters" only: gradient hooks crash with
            # find_unused_parameters=True (None grads reach wandb's callback)
            wandb.watch(self.model, log="parameters", log_freq=100)

        self.log = log
        self.run_path = run_path
        if local_rank == 0:
            self.log.info(f'model_size: {sum(param.nelement() for param in self.model.parameters())}')
            self.log.info(f'number_of_parameters_that_require_gradient: {sum(param.nelement() for param in filter(lambda p: p.requires_grad, self.model.parameters()))}')
            self.log.info(f'torch version: {torch.__version__}')
            self.log.info(f'PIL version: {PIL.__version__}')

        self.train_integrator = Integrator(self.log, distributed=True)

        self.train()
        parameter_groups = get_parameter_groups(self.model, stage_cfg, print_log=(local_rank == 0))

        self.optimizer = optim.AdamW(
            parameter_groups,
            lr=stage_cfg['learning_rate'],
            weight_decay=stage_cfg['weight_decay'],
            eps=1e-6 if self.use_amp else 1e-8,
            foreach=True)

        self.loss_computer = LossComputer(cfg, stage_cfg)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(init_scale=8192)
        self.clip_grad_norm = stage_cfg['clip_grad_norm']

        if stage_cfg['lr_schedule'] == 'constant':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
        elif stage_cfg['lr_schedule'] == 'poly':
            total_num_iter = stage_cfg['num_iterations']
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: (1 - (x / total_num_iter))**0.9)
        elif stage_cfg['lr_schedule'] == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, stage_cfg['lr_schedule_steps'], stage_cfg['lr_schedule_gamma'])
        else:
            raise NotImplementedError

        self.log_text_interval = cfg['log_text_interval']
        self.log_image_interval = cfg['log_image_interval']
        if cfg['debug']:
            self.log_text_interval = self.log_image_interval = 1

        self.log.time_estimator = TimeEstimator(3000, self.log_text_interval)
        self.train_integrator = Integrator(self.log, distributed=True)
        self._is_train = True
        self.clip_grad_norm = stage_cfg['clip_grad_norm']

        # --- UDA Stage 1 (guarded: no cfg.uda / enable=false / no target
        # loader -> exactly the source-only behavior above) ---
        uda_cfg = cfg.get('uda', None)
        self.uda_enable = bool(uda_cfg is not None and target_loader is not None
                               and uda_cfg.get('enable', False))
        # defaults so source-only runs (no uda block) stay compatible with
        # do_pass; _setup_uda overwrites them when UDA is on
        self.proto_enable = False
        self.tc_enable = False
        self.gdr_rel_enable = False
        self.proto_mode = 'original'
        self.proto_align_mode = 'original'
        self.target_loader = target_loader if self.uda_enable else None
        self.target_iter = None
        self._uda_epoch = 0
        if self.uda_enable:
            self._setup_uda(local_rank)

    # ================================================================ UDA S1
    def _setup_uda(self, local_rank):
        uc = self.uda_cfg = self.cfg['uda']
        self.uda_lambda_self = float(uc.get('lambda_self', 0.1))
        self.uda_warmup_iters = int(uc.get('warmup_iters', 500))
        self.uda_conf_thr = float(uc.get('conf_thr', 0.9))
        self.uda_conf_mode = str(uc.get('conf_mode', 'hard'))
        self.uda_frame_conf_thr = float(uc.get('frame_conf_thr', 0.85))
        self.uda_min_pseudo_px = float(uc.get('min_pseudo_px', 50))
        # Stage 2: phase-aware area-curve temporal consistency (gated)
        tc = uc.get('temporal_consistency', None)
        self.tc_enable = bool(tc is not None and tc.get('enable', False))
        if self.tc_enable:
            self.tc_lambda = float(tc.get('lambda_tc', 0.03))
            self.tc_area_bins = int(tc.get('area_bins', 4))
            self.tc_jump_thr = float(tc.get('jump_thr', 0.15))
            self.tc_area_mode = str(tc.get('area_mode', 'sum'))
            self.tc_min_area_px = float(tc.get('min_area_px', 300.0))
        # Stage 3: structure-aware prototype alignment (gated)
        # ---- Reliability-aware GDR (per-frame target reliability r_t) ----
        gr = dict(uc.get('gdr_reliability', None) or {})
        self.gdr_rel_enable = bool(gr.get('enable', False))
        if self.gdr_rel_enable:
            self.gdr_rel_source = str(gr.get('source', 'gdr_readout'))
            self.gdr_rel_gamma = float(gr.get('gamma', 2.0))
            self.gdr_rel_normalize = bool(gr.get('normalize', True))
            self.gdr_rel_clamp = (float(gr.get('clamp_min', 0.3)),
                                  float(gr.get('clamp_max', 1.0)))
            self.gdr_rel_mode = str(gr.get('mode', 'raw'))
            if self.gdr_rel_mode not in ('raw', 'phase_corrected'):
                raise ValueError(f'unknown gdr_reliability.mode: '
                                 f'{self.gdr_rel_mode}')
            self.gdr_rel_lambda_motion = float(gr.get('lambda_motion', 1.0))
            self.gdr_rel_m_max = float(gr.get('m_max', 1.0))
            self.gdr_rel_apply = str(gr.get('apply', 'both'))
            if self.gdr_rel_apply not in ('pseudo', 'proto', 'both'):
                raise ValueError(f'unknown gdr_reliability.apply: '
                                 f'{self.gdr_rel_apply}')
        proto = uc.get('prototype_alignment', None)
        self.proto_enable = bool(proto is not None and proto.get('enable', False))
        self.uda_need_features = self.proto_enable or self.gdr_rel_enable
        if self.proto_enable:
            self.proto_lambda = float(proto.get('lambda_proto', 0.01))
            self.proto_warmup_iters = int(proto.get('proto_warmup_iters', 500))
            self.proto_conf_thr = float(proto.get('proto_conf_thr', 0.9))
            self.proto_boundary_radius = int(proto.get('boundary_radius', 2))
            self.proto_bg_inner_radius = int(proto.get('bg_inner_radius', 2))
            self.proto_bg_outer_radius = int(proto.get('bg_outer_radius', 6))
            self.proto_min_pixels = int(proto.get('min_pixels', 20))
            self.proto_regions = tuple(proto.get('regions', ['fg', 'bd', 'bg']))
            self.proto_region_weights = dict(proto.get('region_weights', {}) or {})
            # graph variant (opt-in; default/original path is untouched)
            self.proto_align_mode = str(proto.get('proto_align_mode', 'original'))
            # which feature space feeds prototype alignment: the encoder
            # 'proto_feat' (default) or the GDR memory readout 'gdr_readout'
            # (returned by the backbone under return_features=True)
            self.proto_feature_key = str(proto.get('feature_key', 'proto_feat'))
            if self.proto_feature_key not in ('proto_feat', 'gdr_readout'):
                raise ValueError(
                    f'unknown prototype_alignment.feature_key: '
                    f'{self.proto_feature_key}')
            # CDPA variant (opt-in; 'mode: contrast' replaces the original
            # raw-feature fg prototype with a contrast-space one)
            self.proto_mode = str(proto.get('mode', 'original'))
            if self.proto_mode not in ('original', 'contrast'):
                raise ValueError(f'unknown prototype_alignment.mode: {self.proto_mode}')
            if self.proto_mode == 'contrast':
                cc = dict(proto.get('contrast', {}) or {})
                self.cdpa_kernel = int(cc.get('kernel_size', 7))
                self.cdpa_contrast_mode = str(cc.get('contrast_mode', 'signed'))
                self.cdpa_normalize = str(cc.get('normalize', 'spatial_z'))
                self.cdpa_use_background = bool(cc.get('use_background', False))
                self.cdpa_bg_weight = float(cc.get('bg_weight', 0.25))
                self.cdpa_min_pixels = int(proto.get('min_pixels', 16))
                # 'true' = mirror the other repo's implementation: only
                # reparameterize the feature space, feed the UNMODIFIED
                # original prototype_alignment_loss (single-variable swap)
                self.cdpa_use_original_pooling = bool(
                    cc.get('use_original_pooling', False))
            if self.proto_align_mode == 'graph':
                self.proto_graph_mem = ProtoGraphMemory(
                    num_segments=int(proto.get('graph_num_segments', 3)),
                    phase_bins=int(proto.get('graph_phase_bins', 4)),
                    dim=int(proto.get('graph_feat_dim', 256)),
                    momentum=float(proto.get('graph_memory_momentum', 0.9)))
                self.proto_graph_num_segments = int(proto.get('graph_num_segments', 3))
                self.proto_graph_attn_dim = int(proto.get('graph_attn_dim', 64))
                self.proto_graph_node_conf_thr = float(proto.get('graph_node_conf_thr', 0.7))
                self.proto_graph_alpha_max = float(proto.get('graph_alpha_max', 0.8))
            elif self.proto_align_mode != 'original':
                raise ValueError(f'unknown proto_align_mode: {self.proto_align_mode}')
            else:
                self.proto_graph_mem = None
        self.teacher = None
        if dict(uc.get('teacher', {})).get('enable', True):
            self.teacher = CFMCTeacher(dict(uc.get('teacher', {})))
            if self.uda_lambda_self > 0:
                self._calibrate_teacher()
            else:
                self.teacher.calibrate(np.zeros(4) + 0.05)  # unused placeholder
        # teacher disabled (ablation A1): raw student prob is the pseudo label
        # target iterator is created lazily on first use: with lambda_self=0
        # the target loader is never touched (source-only RNG parity)
        if local_rank == 0:
            if self.teacher is not None:
                tinfo = (f"teacher=D8b_ab stateful={self.teacher.stateful} "
                         f"ent_p50={self.teacher.entropy_low:.4f} "
                         f"ent_p95={self.teacher.entropy_high:.4f} "
                         f"write_policy={self.teacher.write_policy}")
            else:
                tinfo = "teacher=DISABLED (raw pseudo labels, ablation A1)"
            print(f"[UDA S1] enabled: lambda_self={self.uda_lambda_self} "
                  f"warmup={self.uda_warmup_iters} conf_mode={self.uda_conf_mode} "
                  f"{tinfo}")

    @torch.no_grad()
    def _calibrate_teacher(self):
        """Dataset-percentile entropy calibration of the CFMC teacher on the
        current model's target-domain outputs (eval mode, no BN update)."""
        n_b = int(self.uda_cfg.get('teacher', {}).get('entropy_calib_batches', 16))
        was_training = self.model.training
        self.model.eval()
        ents = []
        it = iter(self.target_loader)
        for _ in range(n_b):
            try:
                batch = next(it)
            except StopIteration:
                break
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda(non_blocking=True)
            out = self.model(batch)
            T = batch['rgb'].shape[1]
            for ti in range(T):
                p = torch.softmax(out[f'logits_{ti}'].float(), dim=1)[:, 1]
                ents.append(frame_entropy(p.cpu().numpy()))
        if was_training:
            self.model.train()
        assert ents, 'UDA calibration found no target batches'
        t = torch.tensor(ents, dtype=torch.float64, device='cuda')
        if dist.get_world_size() > 1:
            gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, t)
            t = torch.cat(gathered)
        lo, hi = self.teacher.calibrate(t.cpu().numpy())
        if self.local_rank == 0:
            print(f'[UDA S1] teacher calibrated on {t.numel()} target frames: '
                  f'ent_p50={lo:.4f} ent_p95={hi:.4f}')

    def _uda_lambda(self, it):
        if self.uda_lambda_self <= 0:
            return 0.0
        return self.uda_lambda_self * min(1.0, max(it, 0) / max(self.uda_warmup_iters, 1))

    def _proto_lambda(self, it):
        if not self.proto_enable or self.proto_lambda <= 0:
            return 0.0
        return self.proto_lambda * min(1.0, max(it, 0) / max(self.proto_warmup_iters, 1))

    def _next_target_batch(self):
        if self.target_iter is None:
            self.target_iter = iter(self.target_loader)
        try:
            return next(self.target_iter)
        except StopIteration:
            if isinstance(self.target_loader.sampler, DistributedSampler):
                self.target_loader.sampler.set_epoch(self._uda_epoch)
            self._uda_epoch += 1
            self.target_iter = iter(self.target_loader)
            return next(self.target_iter)

    def _uda_target_step(self):
        """Second (grad-carrying) forward on a target batch + CFMC-teacher
        pseudo loss. Teacher runs no-grad on detached fg probabilities."""
        batch = self._next_target_batch()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        out = self.model(batch, return_features=self.uda_need_features)
        batch.update(out)
        B, T = batch['rgb'].shape[0], batch['rgb'].shape[1]
        with torch.no_grad():
            # fg prob identical to inference-time masks[:, 1:] semantics
            probs = np.stack([
                torch.softmax(batch[f'logits_{ti}'].float(), dim=1)[:, 1].cpu().numpy()
                for ti in range(T)], axis=1)  # [B,T,H,W]
            if self.teacher is not None:
                tout = self.teacher.process_batch(
                    [str(n) for n in batch['info']['name']], probs)
            else:  # ablation A1: raw pseudo labels, no CFMC stabilization
                p_raw = probs.astype(np.float32)
                tout = {
                    'p_stab': p_raw,
                    'conf_map': np.clip(2.0 * np.abs(p_raw - 0.5), 0, 1),
                    'frame_fg_px': (p_raw > 0.5).sum(axis=(2, 3)),
                    'frame_mean_conf': np.clip(2.0 * np.abs(p_raw - 0.5), 0, 1)
                                        .mean(axis=(2, 3)),
                }
        device = batch['logits_0'].device
        logits_stack = torch.stack([batch[f'logits_{ti}'] for ti in range(T)],
                                   dim=1)  # [B,T,2,H,W]
        p_stab = torch.from_numpy(tout['p_stab']).to(device)
        conf = torch.from_numpy(tout['conf_map']).to(device)
        fmc = torch.from_numpy(tout['frame_mean_conf']).float().to(device)
        ffpx = torch.from_numpy(tout['frame_fg_px']).float().to(device)
        loss, stats = pseudo_label_loss(
            logits_stack, p_stab, conf, fmc, ffpx,
            conf_thr=self.uda_conf_thr, conf_mode=self.uda_conf_mode,
            frame_conf_thr=self.uda_frame_conf_thr,
            min_pseudo_px=self.uda_min_pseudo_px)
        if self.tc_enable:
            tc_loss, tc_stats = area_smooth_loss(
                logits_stack, area_bins=self.tc_area_bins,
                jump_thr=self.tc_jump_thr, area_mode=self.tc_area_mode,
                min_area_px=self.tc_min_area_px)
            stats.update(tc_stats)
            result = {'loss': loss, 'tc_loss': tc_loss, 'stats': stats}
        else:
            result = {'loss': loss, 'stats': stats}
        if self.teacher is not None:
            stats.update(self.teacher.rw_summary())
            stats.update(self.teacher.cc_summary())
            dbg = getattr(self.teacher, 'cc_debug_records', None)
            if dbg:
                for rec in dbg:
                    print(f"[CFMC-CC] {batch['info']['name']} frame {rec['frame']}: "
                          f"n_comp={rec['num_components']} sel_area={rec['selected_area']:.0f} "
                          f"rm={rec['removed_area']:.0f} ({rec['removed_area_ratio']:.2%}) "
                          f"conf_sel={rec['mean_conf_selected']:.3f} "
                          f"conf_rm={rec['mean_conf_removed']:.3f} "
                          f"changed={rec['changed']}")
                dbg.clear()
        if self.proto_enable or self.gdr_rel_enable:
            # Stage 3: hand the target side (features / pseudo mask / conf)
            # to do_pass for prototype alignment
            result['features'] = out.get('features')
            result['pseudo_mask'] = (p_stab > 0.5).float()
            result['conf_map_t'] = conf
        if self.gdr_rel_enable:
            # Reliability-aware GDR: per-frame weights r_t from GDR
            # readout/state dynamics (detached; pure reweighting).
            # phase_corrected mode: divide delta by (1 + lambda_motion*m_t),
            # m_t = normalized prob-fg area jump (genuine fast contraction/
            # relaxation is protected from being down-weighted).
            m_t = None
            if self.gdr_rel_mode == 'phase_corrected':
                area = probs.sum(axis=(2, 3))                # [B,T] prob area
                prev = np.concatenate([area[:, :1], area[:, :-1]], axis=1)
                m = np.abs(area - prev) / (prev + 1e-6)
                m[:, 0] = 0.0                                # no predecessor
                m = np.clip(m, 0.0, self.gdr_rel_m_max)
                m_t = torch.from_numpy(m.astype(np.float32)).to(device)
                stats['rel_m_mean'] = float(m.mean())
                stats['rel_m_std'] = float(m.std())
                stats['rel_m_max'] = float(m.max())
            try:
                from model.uda.gdr_reliability import gdr_reliability_scores
            except ImportError as _e:
                raise ImportError('gdr_reliability module not shipped in this '
                                  'release (research ablation)') from _e
            r_t = gdr_reliability_scores(
                out['features'], self.gdr_rel_source, self.gdr_rel_gamma,
                self.gdr_rel_normalize, *self.gdr_rel_clamp,
                motion=m_t, lambda_motion=(self.gdr_rel_lambda_motion
                                           if m_t is not None else 0.0))
            result['r_t'] = r_t
            stats['rel_mean'] = float(r_t.mean().item())
            stats['rel_std'] = float(r_t.std().item())
            stats['rel_min'] = float(r_t.min().item())
            stats['rel_max'] = float(r_t.max().item())
            if self.gdr_rel_apply in ('pseudo', 'both'):
                # reweight the pseudo loss per frame (mean-normalized
                # weights, so the loss scale matches the baseline)
                loss, rel_stats = reliability_weighted_pseudo_loss(
                    logits_stack, p_stab, conf, fmc, ffpx, r_t,
                    conf_thr=self.uda_conf_thr, conf_mode=self.uda_conf_mode,
                    frame_conf_thr=self.uda_frame_conf_thr,
                    min_pseudo_px=self.uda_min_pseudo_px)
                result['loss'] = loss
                stats.update(rel_stats)
        return result

    def do_pass(self, data, it=0):
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda(non_blocking=True)

        out = self.model(data, return_features=self.proto_enable)

        num_objects = out.get('num_objects', [1] * data['rgb'].shape[0])
        data.update(out)

        T = data['rgb'].shape[1]
        supervised_indices = torch.tensor([0, T - 1], device=data['rgb'].device)
        all_logits_keys = [f'logits_{ti}' for ti in supervised_indices.tolist()]
        
        if not all(k in data for k in all_logits_keys):
            raise KeyError(f"Missing keys in data: {all_logits_keys}")

        new_pred = torch.stack([data[key] for key in all_logits_keys], dim=1)
        new_gt = torch.index_select(data['cls_gt'], 1, supervised_indices)
        data.update({'logits': new_pred, 'masks': new_gt})

        if self.use_amp:
            with torch.amp.autocast(device_type='cuda'):
                losses = self.loss_computer.compute(data, num_objects)
        else:
            losses = self.loss_computer.compute(data, num_objects)
        loss = losses['total_loss']

        # --- UDA Stage 1: L = L_src + lambda_self(t) * L_self ---
        #     Stage 2: + lambda_tc * L_area_smooth
        #     Stage 3: + lambda_proto(t) * L_proto (structure prototypes)
        uda_stats = None
        if self.uda_enable:
            lam = self._uda_lambda(it)
            if lam > 0:
                uda_out = self._uda_target_step()
                loss = loss + lam * uda_out['loss']
                uda_stats = uda_out['stats']
                losses['loss_pseudo'] = uda_out['loss']
                if 'tc_loss' in uda_out:
                    losses['loss_tc'] = uda_out['tc_loss']
                    loss = loss + self.tc_lambda * uda_out['tc_loss']
                if self.proto_enable:
                    proto_lam = self._proto_lambda(it)
                    if proto_lam > 0:
                        src_mask = (data['cls_gt'][:, :, 0] > 0).float()
                        fsel = self.proto_feature_key
                        src_feats = data['features'][fsel]
                        tgt_feats = uda_out['features'][fsel]
                        if self.gdr_rel_enable and \
                                self.gdr_rel_apply in ('proto', 'both') and \
                                'r_t' in uda_out:
                            # reliability-scaled target prototype pooling
                            # weight: pseudo_mask * conf * r_t
                            uda_out['conf_map_t'] = \
                                uda_out['conf_map_t'] * \
                                uda_out['r_t'][:, :, None, None]
                        if self.proto_mode == 'contrast' and \
                                self.cdpa_use_original_pooling:
                            # space-only swap (mirrors the other repo):
                            # reparameterize both feature spaces, keep the
                            # original loss internals untouched
                            def _cs(x):
                                b, t = x.shape[:2]
                                y = contrast_transform(
                                    x.reshape(b * t, *x.shape[2:]),
                                    self.cdpa_kernel, self.cdpa_contrast_mode,
                                    self.cdpa_normalize)
                                return y.reshape(b, t, *y.shape[1:])
                            fs = _cs(src_feats)
                            ft = _cs(tgt_feats)
                            proto_loss, proto_stats = prototype_alignment_loss(
                                fs, ft, src_mask, uda_out['pseudo_mask'],
                                conf_t=uda_out['conf_map_t'],
                                proto_conf_thr=self.proto_conf_thr,
                                boundary_radius=self.proto_boundary_radius,
                                bg_inner_radius=self.proto_bg_inner_radius,
                                bg_outer_radius=self.proto_bg_outer_radius,
                                min_pixels=self.proto_min_pixels,
                                regions=self.proto_regions,
                                region_weights=self.proto_region_weights)
                            loss = loss + proto_lam * proto_loss
                            losses['loss_cdpa'] = proto_loss
                            uda_stats = dict(uda_stats,
                                             proto_mode='cspace_origpool',
                                             proto_lambda=proto_lam,
                                             **proto_stats)
                        elif self.proto_mode == 'contrast':
                            # CDPA: contrast-space fg prototype alignment
                            # (strict replacement of the original raw-feature
                            # fg cosine; same anchors, same lambda)
                            fs = src_feats
                            ft = tgt_feats
                            n = fs.shape[0] * fs.shape[1]
                            proto_loss, proto_stats = \
                                contrast_proto_alignment_loss(
                                    fs.reshape(n, *fs.shape[2:]),
                                    src_mask.reshape(n, *src_mask.shape[2:]),
                                    ft.reshape(n, *ft.shape[2:]),
                                    uda_out['pseudo_mask'].reshape(
                                        n, *uda_out['pseudo_mask'].shape[2:]),
                                    tgt_conf=uda_out['conf_map_t'].reshape(
                                        n, *uda_out['conf_map_t'].shape[2:]),
                                    kernel_size=self.cdpa_kernel,
                                    contrast_mode=self.cdpa_contrast_mode,
                                    normalize=self.cdpa_normalize,
                                    min_pixels=self.cdpa_min_pixels,
                                    use_background=self.cdpa_use_background,
                                    bg_weight=self.cdpa_bg_weight)
                            loss = loss + proto_lam * proto_loss
                            losses['loss_cdpa'] = proto_loss
                            uda_stats = dict(uda_stats,
                                             proto_mode='contrast',
                                             proto_lambda=proto_lam,
                                             **proto_stats)
                        elif self.proto_align_mode == 'graph':
                            from model.uda.proto_align_graph import \
                                prototype_alignment_loss_graph
                            proto_loss, proto_stats = \
                                prototype_alignment_loss_graph(
                                    src_feats,
                                    tgt_feats,
                                    src_mask, uda_out['pseudo_mask'],
                                    conf_t=uda_out['conf_map_t'],
                                    memory=self.proto_graph_mem,
                                    num_segments=self.proto_graph_num_segments,
                                    min_pixels=self.proto_min_pixels,
                                    pool_conf_thr=self.proto_conf_thr,
                                    node_conf_thr=self.proto_graph_node_conf_thr,
                                    attn_dim=self.proto_graph_attn_dim,
                                    alpha_max=self.proto_graph_alpha_max)
                        else:   # original single-node cosine (unchanged)
                            proto_loss, proto_stats = prototype_alignment_loss(
                                src_feats,
                                tgt_feats,
                                src_mask, uda_out['pseudo_mask'],
                                conf_t=uda_out['conf_map_t'],
                                proto_conf_thr=self.proto_conf_thr,
                                boundary_radius=self.proto_boundary_radius,
                                bg_inner_radius=self.proto_bg_inner_radius,
                                bg_outer_radius=self.proto_bg_outer_radius,
                                min_pixels=self.proto_min_pixels,
                                regions=self.proto_regions,
                                region_weights=self.proto_region_weights)
                        loss = loss + proto_lam * proto_loss
                        losses['loss_proto'] = proto_loss
                        uda_stats = dict(uda_stats,
                                         proto_lambda=proto_lam,
                                         **proto_stats)
                uda_stats = dict(uda_stats, uda_lambda=lam)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: NaN or inf detected in loss: {loss}. Skipping batch.")
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        self.scheduler.step()

        loss_val = loss.item()
        self.log.log_scalar('loss', loss_val, it)
        if self.local_rank == 0:
            wandb.log({
                "loss": loss_val,
                "lr": self.scheduler.get_last_lr()[0],
                **{k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}
            }, step=it)

        if self.local_rank == 0 and it % self.log_text_interval == 0 and it != 0:
            lr = self.scheduler.get_last_lr()[0]
            self.log.log_scalar('lr', lr, it)
            if uda_stats is not None:
                _tc = (f" tc={uda_stats['tc_second_diff_valid']:.2f}"
                       if 'tc_second_diff_valid' in uda_stats else '')
                _pr = (f" proto={uda_stats['proto_valid_ratio']:.2f}"
                       f" cos_fg={uda_stats['cosine_fg']:.3f}"
                       f" cos_bd={uda_stats['cosine_bd']:.3f}"
                       f" cos_bg={uda_stats['cosine_bg']:.3f}"
                       if 'proto_valid_ratio' in uda_stats else '')
                _rw = (f" rw_acc={uda_stats['rw_acceptance']:.2f}"
                       if 'rw_acceptance' in uda_stats else '')
                _cd = (f" cdpa={uda_stats['cdpa_loss_fg']:.3f}"
                       if 'cdpa_loss_fg' in uda_stats else '')
                self.log.info(
                    "[UDA] it={it} lam={lam:.4f} valid={vr:.2f} fg={fg:.3f} "
                    "conf={cf:.3f} dice={dc:.4f} bce={bc:.4f}{tc}{pr}{rw}{cd}".format(
                        it=it, lam=uda_stats['uda_lambda'],
                        vr=uda_stats['uda_valid_ratio'],
                        fg=uda_stats['uda_fg_ratio'],
                        cf=uda_stats['uda_mean_conf'],
                        dc=uda_stats['uda_dice'], bc=uda_stats['uda_bce'],
                        tc=_tc, pr=_pr, rw=_rw, cd=_cd))

        if self.local_rank == 0 and uda_stats is not None:
            wandb.log({f'uda/{k}': float(v) for k, v in uda_stats.items()
                       if isinstance(v, (int, float))}, step=it)

        return loss_val

    def _extract_first_last_masks(self, out, batch_idx):
        masks_first = None
        masks_last = None

        if isinstance(out, dict):
            masks_keys = sorted([key for key in out.keys() if key.startswith('masks_')], key=lambda x: int(x.split('_')[-1]))
            if masks_keys:
                first_key, last_key = masks_keys[0], masks_keys[-1]
                masks_first, masks_last = out.get(first_key), out.get(last_key)
            else:
                if self.local_rank == 0: self.log.warning(f"[Val] Batch {batch_idx}: No 'masks_X' keys found in model output.")
        elif isinstance(out, torch.Tensor):
            masks_first = masks_last = out
        else:
            if self.local_rank == 0: self.log.warning(f"[Val] Batch {batch_idx}: Unexpected model output type: {type(out)}")

        if masks_first is not None and masks_first.dim() == 4 and masks_first.shape[1] > 1:
            masks_first = masks_first[:, 1, :, :].unsqueeze(1)
            masks_last = masks_last[:, 1, :, :].unsqueeze(1)
        
        if masks_first is None or masks_last is None:
            self.log.warning(f"[Val] Batch {batch_idx}: 'masks_first' or 'masks_last' not found or shape unexpected.")
            return None, None

        return masks_first, masks_last

    def evaluate(self, val_loader, epoch, local_rank, world_size, run_path, it):
        return self._run_evaluation(data_loader=val_loader, mode='val', epoch=epoch, local_rank=local_rank, world_size=world_size, run_path=run_path, it=it)

    def test(self, test_loader, epoch, local_rank, world_size, run_path, it):
        return self._run_evaluation(data_loader=test_loader, mode='test', epoch=epoch, local_rank=local_rank, world_size=world_size, run_path=run_path, it=it)

    def _run_evaluation(self, data_loader, mode, epoch, local_rank, world_size, run_path, it):
        if self.local_rank == 0:
            self.log.info(f"[{mode.capitalize()}] Iter {it} Epoch {epoch}: Evaluating...")

        self.model.eval()

        global_metrics = {"dice": 0.0, "iou": 0.0}
        total_count = 0

        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                metrics = {"dice": 0.0, "iou": 0.0}
                count = 0

                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.cuda(non_blocking=True)

                out = self.model(batch_data)
                masks_first, masks_last = self._extract_first_last_masks(out, it)
                if masks_first is None or masks_last is None:
                    self.log.warning(f"[{mode.capitalize()}] Iter {it}: Skipping batch due to missing masks.")
                    continue

                gt_BTCHW = batch_data.get('cls_gt')
                if gt_BTCHW is None or gt_BTCHW.dim() != 5 or gt_BTCHW.shape[2] != 1:
                    self.log.warning(f"[{mode.capitalize()}] Iter {it}: 'cls_gt' shape unexpected or missing.")
                    continue
                
                gt = gt_BTCHW.squeeze(2)
                gt_first = gt[:, 0, :, :]
                gt_last = gt[:, -1, :, :]
                pred_first = (masks_first > 0.5).long()
                pred_last = (masks_last > 0.5).long()

                pred_first_np = pred_first.squeeze(1).cpu().numpy()
                pred_last_np = pred_last.squeeze(1).cpu().numpy()
                gt_first_np = gt_first.cpu().numpy()
                gt_last_np = gt_last.cpu().numpy()

                for i in range(pred_first_np.shape[0]):
                    try:
                        # sparse-GT support: skip frames with empty GT masks
                        # (dense-GT datasets are unaffected -- never empty)
                        per_frame = []
                        if gt_first_np[i].sum() > 0:
                            per_frame.append((pred_first_np[i], gt_first_np[i]))
                        if gt_last_np[i].sum() > 0:
                            per_frame.append((pred_last_np[i], gt_last_np[i]))
                        if not per_frame:
                            continue
                        dice_b = np.mean([dice_coefficient(p, g) for p, g in per_frame])
                        iou_b = np.mean([sespiou_coefficient(p, g)[0] for p, g in per_frame])

                        metrics["dice"] += dice_b
                        metrics["iou"] += iou_b
                        count += 1
                    except Exception as e:
                        self.log.error(f"[{mode.capitalize()}] Iter {it}, Sample {i}: Error computing metrics: {e}")
                
                if count > 0:
                    for key in metrics: metrics[key] /= count
                    for k in global_metrics.keys(): global_metrics[k] += metrics[k]
                    total_count += 1

                self.log.debug(f"[Val] Iter {it}: Metrics this batch: {metrics}, Count: {count}")

                if self.local_rank == 0 and batch_idx < self.cfg.eval_stage.num_vis:
                    rgb_seq = batch_data['rgb'][0].cpu().numpy()
                    cls_gt_seq = batch_data['cls_gt'][0].cpu().numpy()
                    visualize_sequence(rgb_seq, cls_gt_seq, out, self.run_path, f"It_{it}_E_{epoch}_{mode}_idx_{batch_idx}")

            if world_size > 1:
                for k in global_metrics.keys():
                    tensor_val = torch.tensor(global_metrics[k], dtype=torch.float32, device='cuda')
                    dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
                    global_metrics[k] = tensor_val.item()

                total_count_t = torch.tensor(total_count, dtype=torch.float32, device='cuda')
                dist.all_reduce(total_count_t, op=dist.ReduceOp.SUM)
                total_count_all = int(total_count_t.item())
            else:
                total_count_all = total_count

            if self.local_rank == 0:
                if total_count_all > 0:
                    for k in global_metrics.keys():
                        global_metrics[k] /= total_count_all
                    
                    log_str = f"[{mode.capitalize()}] Iter={it}, "
                    for k, v in global_metrics.items():
                        log_str += f"{k.upper()}={v:.4f}, "
                    self.log.info(log_str[:-2])
                    
                    wandb_metrics = {f"{mode}_{k}": v for k, v in global_metrics.items()}
                    wandb_metrics["epoch"] = epoch
                    wandb.log(wandb_metrics, step=it)

                    for k, v in global_metrics.items():
                        wandb.summary[f"{mode}_{k}"] = v
                else:
                    self.log.warning(f"[{mode.capitalize()}] No valid samples were evaluated.")

        self.model.train()
        self._freeze_bn_eval(self.model)
        return global_metrics

    def save_weights(self, it):
        if self.local_rank != 0: return
        os.makedirs(self.run_path, exist_ok=True)
        save_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_{it}.pth')
        torch.save(self.model.module.state_dict(), save_path)
        self.log.info(f'[Trainer] Weights saved: {save_path}')

    def save_checkpoint(self, it):
        if self.local_rank != 0: return
        os.makedirs(self.run_path, exist_ok=True)
        ckpt = {
            'it': it,
            'weights': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        save_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_ckpt_{it}.pth')
        torch.save(ckpt, save_path)
        self.log.info(f'[Trainer] Checkpoint saved: {save_path}')

    def load_checkpoint(self, path):
        map_loc = f'cuda:{self.local_rank}'
        checkpoint = torch.load(path, map_location=map_loc)
        it = checkpoint['it']
        self.model.module.load_state_dict(checkpoint['weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.log.info(f'[Trainer] Loaded checkpoint from {path}, it={it}')
        return it

    def load_weights(self, path):
        map_loc = f'cuda:{self.local_rank}'
        weights = torch.load(path, map_location=map_loc)
        self.model.module.load_state_dict(weights)
        self.log.info(f'[Trainer] Loaded weights from {path}')

    def _freeze_bn_eval(self, model):
        """Keep BatchNorm in eval mode when only the memory-cache routing
        projections train, so running stats stay frozen."""
        if not getattr(self, 'freeze_memory_cache_only', False):
            return
        core = model.module if hasattr(model, 'module') else model
        for m in core.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self):
        self._is_train = True
        self.model.train()
        self._freeze_bn_eval(self.model)
        return self

    def val(self):
        self._is_train = False
        self.model.eval()
        return self
