import os, copy
import numpy as np
import torch, einops, wandb, pdb

from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.timer import Timer
from diffuser.utils.train_utils import get_lr
from diffuser.utils.training import cycle, EMA
import diffuser.utils as utils
from diffuser.guides.render_m2d import Maze2dRenderer_V2


class OgB_InvDyn_Trainer_v1(object):
    def __init__(
        self,
        inv_model: torch.nn.Module,
        dataset,
        renderer,
        goal_sel_idxs, ## which dims in obs/stateSpace will be used as goal
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,

        results_folder='./results',
        n_reference=8,
        n_samples=2,
        device='cuda',
        trainer_dict={},
    ):
        super().__init__()
        self.model = inv_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq


        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=6, shuffle=True, pin_memory=True
        ))
        self.goal_sel_idxs = goal_sel_idxs

        ## --- goal-horizon policy (re-posed inverse dynamics) ---------------------
        ## Default 'rand' reproduces the ORIGINAL behaviour exactly: the goal LiDAR
        ## scan is sampled at a RANDOM future step in [1, horizon). That target is
        ## ill-posed (one (state, goal-scan) -> many valid first actions, so MSE
        ## floors at the action variance; see LIDAR_DIAGNOSIS_AND_BASELINE.md).
        ## Set goal_horizon_mode='fixed' (via the config's trainer_dict) for a
        ## well-posed k-step-ahead inverse dynamics.
        self.trainer_dict = trainer_dict or {}
        self.goal_horizon_mode = self.trainer_dict.get('goal_horizon_mode', 'rand')
        self.fixed_goal_k = int(self.trainer_dict.get('fixed_goal_k', 1))
        self.goal_k_max = self.trainer_dict.get('goal_k_max', None)  # None => exactly k; else random [k, k_max]
        ## --- v2 goal-aware levers (2026-07-07; defaults reproduce legacy bit-exact) ---
        ## goal_encoding: 'abs' (legacy) | 'diff' -> the model's goal input becomes
        ## normed(goal scan) - normed(current scan). "Stay" is then the ZERO vector
        ## (an intrinsic stop signal) and rotation/translation appear as sign
        ## patterns the MLP cannot ignore as easily as an absolute scan pair
        ## (the original xy-goal inv-dyn steered because direction was trivially
        ## decodable; this restores that decodability in scan space).
        self.goal_encoding = str(self.trainer_dict.get('goal_encoding', 'abs'))
        assert self.goal_encoding in ('abs', 'diff'), self.goal_encoding
        ## act_chunk: 1 (legacy, predict the first action) | m>1 -> predict the
        ## next m actions (flattened, masked L2). The chunk tail contains the
        ## turn toward the goal, so I(goal; a_{1:m} | state) >> I(goal; a_1 |
        ## state) ~= 0 -- the loss finally FORCES goal usage (the single-action
        ## target lets momentum explain a_1, which is why h12's goal channel is
        ## a 20% whisper and wide-k went fully goal-blind).
        self.act_chunk = int(self.trainer_dict.get('act_chunk', 1) or 1)
        utils.print_color(
            f'[invdyn trainer] goal_horizon_mode={self.goal_horizon_mode} '
            f'fixed_goal_k={self.fixed_goal_k} goal_k_max={self.goal_k_max} '
            f'goal_encoding={self.goal_encoding} act_chunk={self.act_chunk}', c='c')

        # pdb.set_trace()
        
        self.renderer: Maze2dRenderer_V2 = renderer

        self.optimizer = torch.optim.Adam(inv_model.parameters(), lr=train_lr)

        self.logdir = results_folder

        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0
        self.device = device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _sample_goal_idxs(self, st_idxs, hzn_tmp, b_s):
        """Per-sample goal index = the future step whose LiDAR scan is the inv-dyn
        goal. Overridable via the config trainer_dict (no subclass needed).

        'rand'  (default): randint(low=st_idxs+1, high=hzn_tmp) -- original behaviour.
        'fixed': a well-posed k-step-ahead goal. With goal_k_max set, draw uniformly
                 in [fixed_goal_k, goal_k_max] (a little robustness to the eval
                 waypoint spacing); otherwise exactly fixed_goal_k. The caller still
                 clips to (val_lens - 1), so short trajectories stay valid.
        """
        if self.goal_horizon_mode == 'rand':
            return np.random.randint(low=st_idxs + 1, high=hzn_tmp)
        elif self.goal_horizon_mode == 'fixed':
            if self.goal_k_max is not None and int(self.goal_k_max) > self.fixed_goal_k:
                lo = max(1, min(self.fixed_goal_k, hzn_tmp - 1))
                hi = max(lo + 1, min(int(self.goal_k_max) + 1, hzn_tmp))  # exclusive upper
                return np.random.randint(low=lo, high=hi, size=b_s)
            k = max(1, min(self.fixed_goal_k, hzn_tmp - 1))
            return np.full(shape=(b_s,), fill_value=k, dtype=np.int64)
        else:
            raise ValueError(f'unknown goal_horizon_mode={self.goal_horizon_mode!r}')

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        self.model.train()

        for i_tr in range(n_train_steps):
            for i_ac in range(self.gradient_accumulate_every):
                ## obs_trajs conditions returns act_trajs is_pads
                batch = next(self.dataloader)
                ## important: check what is inside the batch
                
                # obs_trajs, cond_st, returns, act_trajs, is_pads = batch

                ## val_lens in range of [2,3,4,...,hzn=10]
                obs_trajs, act_trajs, conditions, val_lens = batch


                ## implement the look-ahead for the inv dyn model
                # pdb.set_trace() ## check shape
                ## obs_trajs: torch.Size([128, 4, 11])
                hzn_tmp = obs_trajs.shape[1] ## e.g., 4
                b_s = obs_trajs.shape[0]
                b_idxs = np.arange( b_s )
                

                if True:
                    ## if hzn=4
                    ## low always 0
                    st_idxs = np.zeros(shape=(b_s,), dtype=np.int32)
                    ## low=1, hzn=4 then possible high=1,2,3, shape: (B,)
                    ## re-posed: policy is overridable via trainer_dict (default 'rand')
                    goal_idxs = self._sample_goal_idxs(st_idxs, hzn_tmp, b_s)

                    # pdb.set_trace() ## check shape

                    ## remove invalid padding states in the trajs
                    ## will be int32 or int64
                    goal_idxs = np.clip(goal_idxs, a_min=0, a_max=val_lens-1)

                    # x_t = obs_trajs[b_idxs, st_idxs, :]
                    x_t = obs_trajs[b_idxs, st_idxs, :]
                    x_t_1 = obs_trajs[b_idxs , goal_idxs, :]
                    
                    ## for goal, just pick the idxs of interest
                    # pdb.set_trace()
                    x_t_1 = x_t_1[:, self.goal_sel_idxs]

                    ## v2a: difference-goal encoding (obs windows are already
                    ## normalized by the dataset, so the diff lives in [-2, 2])
                    if self.goal_encoding == 'diff':
                        x_t_1 = x_t_1 - x_t[:, list(self.goal_sel_idxs)]

                    # pdb.set_trace()
                    if self.act_chunk > 1:
                        ## v2b: target = the next m actions, val_len-masked. The
                        ## window is zero-padded past the episode end; mask the
                        ## loss (j < val_len-1: the last obs of a clipped window
                        ## has no in-window successor, so its action is dropped).
                        m = self.act_chunk
                        assert hzn_tmp >= m, (hzn_tmp, m)
                        a_t = act_trajs[:, :m, :].reshape(b_s, -1)   # (B, m*act_dim)
                        val_t = torch.as_tensor(np.asarray(val_lens).reshape(-1),
                                                dtype=torch.long)
                        a_mask = (torch.arange(m)[None, :]
                                  < (val_t[:, None] - 1).clamp(min=1)).float()
                        a_mask = a_mask[:, :, None].expand(
                            b_s, m, act_trajs.shape[-1]).reshape(b_s, -1)
                    else:
                        ## always the first action (legacy, bit-exact)
                        a_t = act_trajs[b_idxs, st_idxs, :]
                        a_mask = None



                assert x_t.shape[0] == b_s

                # obs_trajs, cond_st, returns, act_trajs, is_pads = utils.to_device_tp(*batch, device=self.device)
                x_t, x_t_1, a_t = utils.to_device_tp(x_t, x_t_1, a_t, device=self.device)



                if a_mask is None:
                    loss, infos = self.model.loss(x_t, x_t_1, a_t)
                else:
                    loss, infos = self.model.loss(x_t, x_t_1, a_t,
                                                  mask=a_mask.to(self.device))

                # pdb.set_trace()

                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

                # pdb.set_trace()
                ## save to online
                metrics = {k:v.detach().item() for k, v in infos.items()}
                
                metrics['train/it'] = self.step
                metrics['train/loss'] = loss.detach().item()
                metrics['train/lr'] = get_lr(self.optimizer)
                wandb.log(metrics, step=self.step)


            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.ema_model.eval()

                self.ema_model.train()

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        utils.print_color(f'[ utils/training ] Saved model to {savepath}', c='y')
        

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])




    def load4resume(self, loadpath):
        data = torch.load(loadpath)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.step = data['step']



    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## obs_trajs: (20, 96, ant:29)
        ## get trajectories and condition at t=0 from batch
        obs_trajs = to_np(batch.obs_trajs)

        ## [ batch_size x horizon x observation_dim (4 pos+vel) ]
        normed_observations = obs_trajs # [:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        tmp_path_1 = os.path.join(self.logdir, f'_sample-reference.png')

        ## (20, h=6, 2)
        obs_trajs_2d = self.get_rowcol_obs_trajs(observations[:, :, :2])
        # pdb.set_trace()
        
        self.renderer.composite(tmp_path_1, obs_trajs_2d)
        


    def get_rowcol_obs_trajs(self, obs_trajs):
        ## special handling for xy in ben's dataset
        dset_type = self.dataset.dset_type 
        if 'ogb' in dset_type:
            ## to support ogbench
            from diffuser.datasets.ogb_dset.ogb_utils import ogb_xy_to_ij
            assert obs_trajs.ndim == 3
            # pdb.set_trace()
            obs_trajs = obs_trajs[:, :, :2] ## should be B,H,D
            obs_trajs = ogb_xy_to_ij(self.dataset.env, xy_trajs=obs_trajs)
            
        elif dset_type != 'ours':
            assert 'ben' in dset_type.lower()
            obs_trajs = utils.ben_xy_to_luo_rowcol(dset_type, obs_trajs)
        
        return obs_trajs