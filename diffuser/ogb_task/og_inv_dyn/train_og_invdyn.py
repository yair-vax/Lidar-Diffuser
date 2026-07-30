import sys, os; sys.path.append('./')
## Force EGL rendering (headless NVIDIA GPU servers). osmesa triggers glGetError
## on these machines; EGL uses the GPU directly. Must be set before any mujoco /
## OpenGL import below.
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
import diffuser.utils as utils
import pdb
import torch, wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=4, sci_mode=False)
import numpy as np; np.set_printoptions(precision=3, suppress=True)


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = None
    config: str


args = Parser().parse_args('diffusion')
# args.n_saves = 5 # 20

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# pdb.set_trace()

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.tot_horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    ###
    max_n_episodes=args.max_n_episodes,
    ###
    termination_penalty=args.termination_penalty,
    use_padding=args.use_padding,
    ## put a link to a smaller dataset for debugging purpose
    dset_h5path=getattr(args, 'dset_h5path', None),
    dataset_config=args.dataset_config,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim ## ant: 29
action_dim = dataset.action_dim ## ant: 8

# test_sample = dataset[0]
# test_sample = dataset[198]

# pdb.set_trace() ## check horizon

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#


# pdb.set_trace()

## Dec 25 19:56 PM
## e.g., MLP_InvDyn_OgB_V3
## v2b action-chunk head (2026-07-07): act_chunk=m>1 widens the action head to
## m*act_dim (the model predicts the next m actions, flattened). Default 1 =
## legacy, bit-exact. Must match trainer_dict['act_chunk'] (configs set both
## from one ACT_CHUNK constant).
act_chunk = int(getattr(args, 'act_chunk', 1) or 1)
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    ##
    device=args.device,
    ##
    input_dim=observation_dim+len(args.goal_sel_idxs),
    action_dim=action_dim * act_chunk, ## NEW name
    obs_dim=observation_dim,
    ##
    act_net_config=args.act_net_config,
    inv_m_config=args.inv_m_config,
)
model = model_config()


#############

# pdb.set_trace()

from diffuser.ogb_task.og_inv_dyn.og_invdyn_training import OgB_InvDyn_Trainer_v1
small_trainer_config = utils.Config(
    OgB_InvDyn_Trainer_v1,
    savepath=(args.savepath, 'inv_trainer_config.pkl'),
    ##
    goal_sel_idxs=args.goal_sel_idxs,
    ema_decay=args.ema_decay,
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    # step_start_ema=args.step_start_ema,
    # update_ema_every=args.update_ema_every,

    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),

    results_folder=args.savepath,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    trainer_dict=args.trainer_dict,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

trainer = small_trainer_config(inv_model=model, 
                              dataset=dataset, 
                              renderer=renderer,
                              device=args.device,
                              )

# pdb.set_trace()


if args.trainer_dict.get('do_train_resume', False): # for a sample resume, should be good
    tmp_path = args.trainer_dict['path_resume']
    utils.print_color(f'Resume From: {tmp_path}', c='c')
    trainer.load4resume( tmp_path )

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model, topk=3)

print('Testing forward...', end='\n', flush=True)
## auto convert dtype to float inside
batch = utils.batchify(dataset[0]) # [1,380,2]
batch = utils.batch_copy(batch, 4)

obs_trajs, act_trajs, _, val_lens = batch

print(f'{obs_trajs.shape=}, {act_trajs.shape=} {val_lens}', flush=True)
# is_pads = is_pads.to(torch.bool)

## obs_trajs.shape torch.Size([4, 3, 11])
x_t = obs_trajs[:,0,:]
# x_t_1 = obs_trajs[:, 2, args.goal_sel_idxs]
## B,obs_dim=29
x_t_1 = obs_trajs[:,1,:]
# pdb.set_trace()
## e.g., B,29 -> B, 2
x_t_1 = x_t_1[ :, args.goal_sel_idxs ]
# pdb.set_trace()
if act_chunk > 1:
    ## chunk head: flattened next-m-actions target (mask exercised in training)
    a_t = act_trajs[:, :act_chunk, :].reshape(act_trajs.shape[0], -1)
else:
    ## B,8
    a_t = act_trajs[:,0,:]

loss, _ = model.loss(x_t, x_t_1, a_t)

loss.backward()

# pdb.set_trace()

print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- save config ---------------------------------#
#-----------------------------------------------------------------------------#


all_configs = dict(dataset_config=dataset_config._dict, 
                render_config=render_config._dict,
                model_config=model_config._dict,
                small_trainer_config=small_trainer_config._dict
                )

# print(args)
ckp_path = args.savepath
## default offline: compute nodes have no tty/api key, and training must never
## die on logging (an 'api_key not configured (no-tty)' crash killed the first
## h64_widek launch when the WANDB_MODE=offline line didn't reach the node).
wandb_mode = os.environ.get('WANDB_MODE', 'offline') if dataset_config.dset_h5path is None else 'disabled'
try:
    wandb.init(
        project="hierarchy-diffuser",
        name=args.logger_name,
        id=args.logger_id,
        dir=ckp_path,
        config=all_configs, ## need to be a dict
        # resume="must",
        mode=wandb_mode,
    )
except Exception as e:
    utils.print_color(f'[wandb] init failed ({e!r}) -> falling back to mode=disabled', c='r')
    wandb.init(mode='disabled')
# pdb.set_trace()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

