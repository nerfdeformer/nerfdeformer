
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import imageio

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.ASpanFormer.aspanformer import ASpanFormer
from src.ASpanFormer.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.aspan_loss import ASpanLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,compute_symmetrical_epipolar_errors_offset_bidirectional,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures,make_matching_figures_offset
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

class PL_ASpanFormer(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['aspan'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = ASpanFormer(config=_config['aspan'])
        self.loss = ASpanLoss(_config)

        # Pretrained weights
        print(pretrained_ckpt)
        if pretrained_ckpt:
            print('load')
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=False)
            print(msg)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')
        
        n_gpu = 4
        for pg in optimizer.param_groups:
            pg['lr'] = 1.25e-4 * n_gpu / 32
        print('LR', [pg['lr'] for pg in optimizer.param_groups])
        print('true_LR', self.config.TRAINER.TRUE_LR, self.trainer.global_step, warmup_step, self.config.TRAINER.CANONICAL_LR)
        # for pg in optimizer.param_groups:
        #     pg['lr'] = 0
        # self.trainer.global_step += 1000
        # input()
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        # print('batch keys 0', list(batch.keys()))
        import time
        t = [time.time()]
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        # print('batch keys 1', list(batch.keys()))
        t.append(time.time())
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        # print('batch keys 2', list(batch.keys()))
        t.append(time.time())
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config) 
        # print('batch keys 3', list(batch.keys()))
        t.append(time.time())
        
        with self.profiler.profile("Compute losses"):
            self.loss(batch) 
        # print('batch keys 4', list(batch.keys()))
        # input()
        t.append(time.time())
        print('time', np.diff(t))
        
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_symmetrical_epipolar_errors_offset_bidirectional(batch) # compute epi_errs for offset match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'epi_errs_offset': [batch['epi_errs_offset_left'][batch['offset_bids_left'] == b].cpu().numpy() for b in range(bs)], #only consider left side
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
   
    def training_step(self, batch, batch_idx):
        # if self.trainer.global_step == 10:
        #     print('set random seed', self.trainer.global_rank)        
        #     set_seed(self.trainer.global_rank)
        # if self.trainer.global_step <= 900:
        #     return {'loss': torch.Tensor([0.]).cuda()}
        self._trainval_inference(batch)
        sch = self.lr_schedulers()
        sch.step()
        # for i in range(100):
        #     sch.step()
        # logging
        if 0 and self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                if not k.startswith('loss_flow') and not k.startswith('conf_'):
                    self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)
            
            #log offset_loss and conf for each layer and level
            layer_num=self.loftr_cfg['coarse']['layer_num']
            for layer_index in range(layer_num):
                log_title='layer_'+str(layer_index)
                self.logger.experiment.add_scalar(log_title+'/offset_loss', batch['loss_scalars']['loss_flow_'+str(layer_index)], self.global_step)
                self.logger.experiment.add_scalar(log_title+'/conf_', batch['loss_scalars']['conf_'+str(layer_index)],self.global_step)
            
            # net-params
            if self.config.ASPAN.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                extra_results = {}
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE, extra_results)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
                    self.logger.experiment.add_scalar(
                        'train_match/n_match_rate', extra_results['n_match'] / extra_results['n_match_gt'],
                        global_step=self.global_step)
                    self.logger.experiment.add_scalar(
                        'train_match/n_match_gt', extra_results['n_match_gt'],
                        global_step=self.global_step)
                #plot offset 
                if self.global_step%200==0:
                    compute_symmetrical_epipolar_errors_offset_bidirectional(batch)
                    figures_left = make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,side='_left')
                    figures_right = make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,side='_right')
                    for k, v in figures_left.items():
                        self.logger.experiment.add_figure(f'train_offset/{k}'+'_left', v, self.global_step)
                    figures = make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,side='_right')
                    for k, v in figures_right.items():
                        self.logger.experiment.add_figure(f'train_offset/{k}'+'_right', v, self.global_step)
        
        # print('aha', list(batch.keys()))
        # if loss has None
        print('training', self.trainer.global_step, batch['idx'], batch['pair_names'], batch['loss'])
        if torch.isnan(batch['loss']).any():
            print('NaN detected', batch['pair_names'])
            exit()
        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        print('val id', batch["idx_val"].cpu().item())
        extra_results = {}
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE, extra_results, 
                                        path = f"/home/zt15/projects/ml-aspanformer/imgs_debug3/{batch['idx_val'].cpu().item()}_{self.global_step}.png",
                                        cv2 = True)
        self.logger.experiment.add_scalar(
            'val_match/n_match_rate', extra_results['n_match'] / extra_results['n_match_gt'],
            global_step=self.global_step)
        self.logger.experiment.add_scalar(
            'val_match/n_match_gt', extra_results['n_match_gt'],
            global_step=self.global_step)
        # for k, v in figures.items():
        #     print('val img', np.array(v).shape)
        #     imageio.imwrite(, v)
            # self.logger.experiment.add_figure(f'val_match/{k}_{batch["idx_val"].cpu().item()}', v, self.global_step)
        return

        ret_dict, _ = self._compute_metrics(batch) #this func also compute the epi_errors
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        figures_offset = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)
            figures_offset=make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,'_left')
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
            'figures_offset_left':figures_offset
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        return
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
