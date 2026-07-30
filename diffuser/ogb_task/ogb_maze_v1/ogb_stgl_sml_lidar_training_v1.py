"""
LiDAR-aware trainer for the CompDiffuser planner.

The base trainer (``OgB_Stgl_Sml_Trainer_v1``) periodically renders sampled
trajectories by interpreting ``obs[..., :2]`` as the agent (x, y) and drawing it on
the maze. With LiDAR observations there is no (x, y) to draw -- the observation is
a vector of beam distances -- so we override the two rendering hooks to instead
visualize the LiDAR scans as (time x beam) heatmaps. Everything else (the
optimization loop, EMA, checkpointing) is inherited unchanged.

All visualization is wrapped in try/except: a plotting hiccup must never kill a
multi-day training run.
"""

import os
import einops
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffuser.utils.arrays import to_np, to_device, apply_dict
from diffuser.models.helpers import apply_conditioning
import diffuser.utils as utils
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_training_v1 import OgB_Stgl_Sml_Trainer_v1


class OgB_Stgl_Sml_Lidar_Trainer_v1(OgB_Stgl_Sml_Trainer_v1):

    # -- helpers -----------------------------------------------------------#
    def _unnorm_obs(self, normed_obs):
        return self.dataset.normalizer.unnormalize(to_np(normed_obs), 'observations')

    def _plot_lidar_grid(self, savepath, panels, titles):
        """panels: list of (H, n_beams) arrays. Saves a row of heatmaps sharing
        a common color scale so generated vs reference scans are comparable."""
        n = len(panels)
        if n == 0:
            return
        vmin = float(min(p.min() for p in panels))
        vmax = float(max(p.max() for p in panels))
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.0), squeeze=False)
        for k, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[0][k]
            im = ax.imshow(panel, aspect='auto', origin='lower',
                           cmap='viridis', vmin=vmin, vmax=vmax,
                           interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('beam')
            if k == 0:
                ax.set_ylabel('time step')
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, label='distance')
        fig.suptitle(f'LiDAR scans (step {self.step})', fontsize=11)
        fig.savefig(savepath, dpi=110, bbox_inches='tight')
        plt.close(fig)

    # -- overridden hooks --------------------------------------------------#
    def render_reference(self, batch_size=10):
        try:
            from diffuser.utils.training import cycle
            dl = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, num_workers=0, shuffle=True))
            batch = dl.__next__()
            dl.close()
            obs = self._unnorm_obs(batch.obs_trajs)  # (B, H, n_beams)
            n_show = min(4, obs.shape[0])
            panels = [obs[i] for i in range(n_show)]
            titles = [f'ref #{i}' for i in range(n_show)]
            savepath = os.path.join(self.logdir, '_sample-reference-lidar.png')
            self._plot_lidar_grid(savepath, panels, titles)
            utils.print_color(f'[lidar trainer] saved {savepath}', c='g')
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[lidar trainer] render_reference skipped: {e}', c='y')

    def _generate_samples(self, batch, n_samples, do_cond):
        """Mirror the base trainer's conditional/unconditional sampling, but
        return raw (normed) samples instead of drawing them on a maze."""
        device = self.device
        stgl_cond = to_device(batch.conditions, device)
        stgl_cond = apply_dict(
            einops.repeat, stgl_cond, 'b d -> (repeat b) d', repeat=n_samples)
        traj_full = einops.repeat(
            batch.obs_trajs, 'b h d -> (repeat b) h d', repeat=n_samples).to(device)

        if do_cond in [None, False]:
            samples = self.ema_model.sample_unCond(batch_size=len(stgl_cond[0]))
        elif do_cond == 'both_ovlp':
            g_cond = dict(do_cond='both_ovlp', traj_full=traj_full, t_type='rand')
            samples = self.ema_model.conditional_sample(g_cond=g_cond)
        elif do_cond == 'both_stgl':
            g_cond = dict(do_cond='both_stgl', traj_full=traj_full, t_type='rand',
                          stgl_cond=stgl_cond)
            samples = self.ema_model.conditional_sample(g_cond=g_cond)
            samples = apply_conditioning(samples, stgl_cond, 0)
        elif do_cond == 'st_endovlp':
            g_cond = dict(do_cond='st_endovlp', traj_full=traj_full, t_type='rand',
                          stgl_cond=stgl_cond)
            samples = self.ema_model.conditional_sample(g_cond=g_cond)
        elif do_cond == 'stovlp_gl':
            g_cond = dict(do_cond='stovlp_gl', traj_full=traj_full, t_type='rand',
                          stgl_cond=stgl_cond)
            samples = self.ema_model.conditional_sample(g_cond=g_cond)
        else:
            raise NotImplementedError(do_cond)
        return samples, traj_full

    def render_samples(self, batch_size=1, n_samples=2, do_cond=None):
        try:
            batch = self.dataloader_vis.__next__()
            samples, traj_full = self._generate_samples(batch, n_samples, do_cond)

            samples = self._unnorm_obs(samples)         # (n_samples, H, n_beams)
            reference = self._unnorm_obs(traj_full)      # (n_samples, H, n_beams)

            n_show = min(3, samples.shape[0])
            panels, titles = [], []
            for i in range(n_show):
                panels.append(reference[i]); titles.append(f'ref #{i}')
                panels.append(samples[i]);   titles.append(f'gen #{i}')

            sample_savedir = self.get_sample_savedir(self.step)
            savepath = os.path.join(
                sample_savedir, f'sample-lidar-{self.step}-{do_cond}.png')
            self._plot_lidar_grid(savepath, panels, titles)
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[lidar trainer] render_samples({do_cond}) skipped: {e}', c='y')