"""
Finetunes PACS.
"""

import sys

sys.path.append('../../')
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
import numpy as onp
from pretrain.dataloader import input_fn_builder, MASK, encoder, AUDIOSPAN
from finetune.common_dataloader import finetune_input_fn_builder, finetune_val_input_fn_builder
from mreserve.modeling import MerlotReserve

from flax.training import train_state
from flax import jax_utils
import flax.linen as nn
from finetune.optimization import construct_finetuning_train_state, finetune_train_step
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32, f32_to_bf16
import argparse
import pandas as pd
import numpy as np
from flax.core.frozen_dict import freeze
from copy import deepcopy
import clu.parameter_overview
import functools
import time
import os

jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')

parser.add_argument(
    'pretrain_config_file',
    help='Where the config.yaml is located',
    type=str,
)
parser.add_argument(
    'ckpt',
    help='checkpoint to use',
    type=str,
)
parser.add_argument(
    '-lr',
    help='lr',
    type=float,
)
parser.add_argument(
    '-ne',
    help='ne',
    type=int,
    default=15,
)
parser.add_argument(
    '-wd',
    help='weight decay',
    type=float,
    default=0.000001,
)
parser.add_argument(
    '-output_grid_h',
    help='output_grid_h',
    type=int,
    default=12,
)
parser.add_argument(
    '-output_grid_w',
    help='output_grid_w',
    type=int,
    default=20,
)
parser.add_argument(
    '-output_name',
    help='output_name',
    type=str,
    default='',
)
parser.add_argument(
    '-wandb_name',
    help='wandb_name',
    type=str,
    default='merlotreserve-pacs',
)
parser.add_argument(
    '-val_batch_size',
    help='val_batch_size -- defaults to 32',
    type=int,
    default=8
)
parser.add_argument(
    '-scan_minibatch',
    help='scan_minibatch -- basically, if this is true then batch size is 1 but we do gradient accumulation',
    action='store_true',
    default=False,
)
args = parser.parse_args()

# # # print(f"Loading from {args.config_file}", flush=True)
with open(args.pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

config['data']['train_fns'] = "YOUR_STORAGE/train{:03d}of008.tfrecord"
config['data']['num_train_files'] = 8
config['data']['random_scale_max'] = 1.0
config['data']['random_scale_min'] = 0.7
config['data']['do_horizontal_flip'] = True

# config['device']['batch_size'] = 16
config['device']['prefetch_size'] = 0
config['device']['n_fns_per_cycle'] = 8

NUM_EPOCH = args.ne
TRAIN_SIZE = 11044
steps_per_epoch = TRAIN_SIZE // config['device']['batch_size']
config['optimizer'] = {
    'beta_2': 0.98,
    'eps': 1e-6,
    'learning_rate': args.lr,
    'num_train_steps': NUM_EPOCH * steps_per_epoch,
    'num_warmup_steps': int(0.5 * steps_per_epoch),
    'use_bfloat16_adam': True,
    'do_bias_correction': True,
    'weight_decay_rate': args.wd,
}

config['data']['swap'] = False
config['device']['iterations_per_loop'] = steps_per_epoch
config['data']['lang_seq_len'] = 256
config['data']['is_train'] = True

cfg_name = args.pretrain_config_file.split('/')[-1]
seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

config['device']['output_dir'] = f'YOUR_STORAGE/checkpoints/{cfg_name}/'
if args.output_name != '':
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], args.output_name)
config['device']['output_dir'] = os.path.join(config['device']['output_dir'], seattle_time)

np.random.seed(1234)
config['model']['output_grid'] = [args.output_grid_h, args.output_grid_w]
ds_train_iter = finetune_input_fn_builder(config, 'pacs')

config['_ckpt'] = args.ckpt
tags = [cfg_name]
if args.output_name != '':
    tags.append(args.output_name)
if (jax.process_index() == 0):
    import wandb
    wandb.init(config=config, project="YOUR_PROJECT", entity='YOUR_ENTITY', notes=f'Loaded from {cfg_name}', tags=tags)
else:
    wandb = None

import shutil
if wandb:
    folder = wandb.run.name
    os.makedirs(f"hist/{folder}", exist_ok=True)
    shutil.copy("../../pretrain/configs/base.yaml", f"hist/{folder}")
    shutil.copy("../../pretrain/configs/large.yaml", f"hist/{folder}")
    shutil.copy("pacs_finetune_audio.py", f"hist/{folder}")
    shutil.copy("pacs_finetune.py", f"hist/{folder}")
    shutil.copy("../common_dataloader.py", f"hist/{folder}")

class MerlotReservePACS(MerlotReserve):
    def setup(self):
        super().setup()
        self.proj = nn.Dense(features=1, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=0.02), name='proj',
                             use_bias=False)

    def __call__(self, batch):
        
        batch_size, images_per_batch, seq_size, img_dim = batch['frames1'].shape

        img_inp1 = jnp.concatenate([batch['midframe1'].reshape(batch_size, 1, seq_size, img_dim), batch['frames1']], axis=1)

        img_inp2 = jnp.concatenate([batch['midframe2'].reshape(batch_size, 1, seq_size, img_dim), batch['frames2']], axis=1)

        frames_enc1 = self.vision_encoder(img_inp1.reshape(batch_size * (images_per_batch+1), seq_size, img_dim))['seq_attnpool']
        frames_enc2 = self.vision_encoder(img_inp2.reshape(batch_size * (images_per_batch+1), seq_size, img_dim))['seq_attnpool']

        frames_enc1 = frames_enc1.reshape(batch_size, (images_per_batch + 1) * seq_size // 4, self.hidden_size)
        frames_enc2 = frames_enc2.reshape(batch_size, (images_per_batch + 1) * seq_size // 4, self.hidden_size)

        text_toks = batch['question']

        text_toks = text_toks.reshape(batch_size, -1)
        subseg_idxs = jnp.zeros((batch_size, text_toks.shape[1]), dtype=jnp.bfloat16)

        inputs1 = self.prepare_multimodal_inputs(
            tokens=text_toks,
            token_segment_idx=subseg_idxs,
            vision_input=frames_enc1,
        )
        inputs2 = self.prepare_multimodal_inputs(
            tokens=text_toks,
            token_segment_idx=subseg_idxs,
            vision_input=frames_enc2,
        )

        joint_enc1 = self.joint_transformer(**inputs1)['seq']
        joint_enc2 = self.joint_transformer(**inputs2)['seq']

        print("je1", joint_enc1.shape)

        pool_idx = jnp.argmax((text_toks == MASK).astype(jnp.float32), 1)
        print("pool_idx", pool_idx)

        pooled_h1 = joint_enc1[jnp.arange(batch_size), pool_idx]

        print("pooled_h1", pooled_h1.shape)
        
        pooled_h2 = joint_enc2[jnp.arange(batch_size), pool_idx]

        logits1 = self.proj(pooled_h1)
        logits2 = self.proj(pooled_h2)

        logits = jnp.concatenate([logits1, logits2], axis=1)
        return logits


model = MerlotReservePACS.from_config(config)
# # print("loaded model")

params = load_checkpoint(args.ckpt)['params']

# Don't need those
for k in ['head', 'span_encoder', 'audio_encoder']:
    params.pop(k, None)
hsz = params['joint_transformer']['final_ln']['bias'].shape[0]
params['proj'] = {'kernel': np.random.randn(hsz, 1).astype(np.float32) * 0.01}
params = freeze(params)

state, tx_fns = construct_finetuning_train_state(opt_config=config['optimizer'], model=model, params=params)

def train_loss_fn(state, params, batch):
    logits = state.apply_fn({'params': params}, batch)
    # print('logits', onp.asarray(logits), flush=True)
    probs = jax.nn.log_softmax(logits, axis=1)
    # print('probs', probs.values, flush=True)

    labels_oh = jax.nn.one_hot(batch['label'],
                               dtype=logits.dtype,
                               num_classes=2)

    # print('labelsoh', labels_oh.values, flush=True)

    loss = -jnp.mean(jnp.sum(labels_oh * probs, axis=-1))
    is_right = (jnp.argmax(probs, axis=1) == batch['label']).astype(jnp.float32).mean()

    return loss, {"loss": loss, "is_right": is_right}

p_train_step = jax.pmap(functools.partial(finetune_train_step, loss_fn=train_loss_fn, tx_fns=tx_fns, scan_minibatch=args.scan_minibatch),
                                          axis_name='batch', donate_argnums=(0,1))

def pred_step(state: train_state.TrainState, batch):
    logits = state.apply_fn({'params': state.params}, batch)
    # print('logits', logits.shape)
    return {'logprobs': jax.nn.log_softmax(logits), 'preds': jnp.argmax(logits, -1)}

p_pred_step = jax.pmap(pred_step, axis_name='batch', donate_argnums=(1,))
# p_pred_step = pred_step

# # print("after pred step")

def val_epoch(state: train_state.TrainState, fns):
    """
    perform a validation epoch
    :param state:
    :return:
    """
    val_config = deepcopy(config)
    val_config['data']['val_fns'] = fns
    val_config['data']['is_train'] = False
    val_config['data']['num_val_files'] = 1
    val_config['data']['do_random_scale'] = False
    val_config['data']['do_horizontal_flip'] = False
    val_config['data']['batch_size'] = args.val_batch_size

    val_iter = finetune_val_input_fn_builder(val_config, 'pacs')

    preds = []

    for ids, batch in val_iter:
        val_pred = p_pred_step(state, batch)

        labels = batch['label'].reshape(-1)
        # print('labels', labels)
        # print('preds', val_pred['preds'].reshape(-1))
        # print('ids', ids)
        for p_i, id_i, label_i in zip(val_pred['preds'].reshape(-1), ids, labels):
            if id_i == 'pad':
                continue
            preds.append({'pred': p_i, 'label': label_i, 'id': id_i})

    preds = pd.DataFrame(preds)
    preds['is_right'] = preds['pred'] == preds['label']
    acc = preds['is_right'].mean()
    return {'acc': acc}

train_metrics = []
# log_every = config['device'].get('commit_every_nsteps', 100)
log_every = 50
time_elapsed = []

best_val = 0
epoch = 0
# the + 1 is because for some reason it crashes at the end otherwise. why? idk/
for n in range(config['optimizer']['num_train_steps']+100):
    st = time.time()
    id_, batch = next(ds_train_iter)
    state, loss_info = p_train_step(state, batch)

    if jax.process_index() == 0:
        train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
        jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])

        step_for_logging = n - log_every
        if step_for_logging >= 0:
            train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}
            wandb.log(train_metrics[step_for_logging], step=step_for_logging, commit=(n + 1) % log_every == 0)

        if (n + 1) % config['device']['iterations_per_loop'] == 0:
            print("Done 1 epoch", flush=True)
            epoch += 1            
            val_info = val_epoch(state, "YOUR_STORAGE/val000of001.tfrecord")
            print(f"Val results: {val_info}", flush=True)
            if val_info['acc'] > best_val:
                print(f"Saving @iter {n:03d}\n", flush=True)
                save_checkpoint(state, path=config['device']['output_dir'], no_optimizer=True)
                best_val = val_info['acc']
            if wandb is not None:
                wandb.log({k + '_test': v for k, v in val_info.items()}, step=step_for_logging, commit=True)

        time_elapsed.append(time.time() - st)
        if len(time_elapsed) >= 100:
            tsum = sum(time_elapsed)
            print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0 / tsum), flush=True)
            time_elapsed = []
    else:
        print("wtf")
