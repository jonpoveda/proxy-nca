import logging, imp

from torchvision.transforms import transforms

import dataset
import utils
import proxynca
import net

import os

os.putenv("OMP_NUM_THREADS", "8")

import torch
import numpy as np
import matplotlib

matplotlib.use('agg', warn=False, force=True)
import matplotlib.pyplot as plt

from datetime import datetime
import time
import argparse
import json
import random
import utils

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

parser = argparse.ArgumentParser(description='Training inception V2' +
                                             ' (BNInception) on CUB200 with Proxy-NCA loss as described in ' +
                                             '`No Fuss Distance Metric Learning using Proxies.`'
                                 )
# export directory, training and val datasets, test datasets
parser.add_argument('--dataset',
                    default='cub',
                    help='Path to root CUB folder, containing the images folder.'
                    )
parser.add_argument('--config',
                    default='config.json',
                    help='Path to root CUB folder, containing the images folder.'
                    )
parser.add_argument('--embedding-size', default=64, type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to InceptionV2.'
                    )
parser.add_argument('--batch-size', default=32, type=int,
                    dest='sz_batch',
                    help='Number of samples per batch.'
                    )
parser.add_argument('--epochs', default=70, type=int,
                    dest='nb_epochs',
                    help='Number of training epochs.'
                    )
parser.add_argument('--log-filename', default='example',
                    help='Name of log file.'
                    )
parser.add_argument('--gpu-id', default=8, type=int,
                    help='ID of GPU that is used for training.'
                    )
parser.add_argument('--workers', default=16, type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.'
                    )

args = parser.parse_args()
torch.cuda.set_device(args.gpu_id)

config = utils.load_config(args.config)
from utils import JSONEncoder, json_dumps

print(json_dumps(obj=config, indent=4, cls=JSONEncoder, sort_keys=True))
with open('log/' + args.log_filename + '.json', 'w') as x:
    json.dump(
        obj=config, fp=x, indent=4, cls=JSONEncoder, sort_keys=True
    )


def save_model(suffix):
    model_path = 'model/{}_{}.pt'.format(args.log_filename, suffix)
    torch.save(model.state_dict(), model_path)


def try_n_times(fn, ntimes, suffix):
    if suffix is None:
        suffix = datetime.now().strftime("%Y%m%d%H%M%S")

    for n in range(ntimes):
        try:
            print('Trying: {}'.format(fn))
            result = fn(suffix)
            print('Done :)')
            return result
        except Exception:
            pass
    print("I've tried! but I could not do it :(")


dl_tr = torch.utils.data.DataLoader(
    dataset.load(
        name=args.dataset,
        root=config['dataset'][args.dataset]['root'],
        classes=config['dataset'][args.dataset]['classes']['train'],
        transform=dataset.utils.make_transform(
            **config['transform_parameters']
        )
    ),
    batch_size=args.sz_batch,
    shuffle=True,
    num_workers=args.nb_workers,
    drop_last=True,
    pin_memory=True
)

dl_ev = torch.utils.data.DataLoader(
    dataset.load(
        name=args.dataset,
        root=config['dataset'][args.dataset]['root'],
        classes=config['dataset'][args.dataset]['classes']['eval'],
        transform=dataset.utils.make_transform(
            **config['transform_parameters'],
            is_train=False
        )
    ),
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
    pin_memory=True
)

# Remove normalization
dl_tr.dataset.transform.transforms[-1] = transforms.Normalize(
    mean=[0, 0, 0], std=[1, 1, 1]
)
print(
    'Transformations are overwritten as:\n{}'.format(dl_tr.dataset.transform))

# Compute statistics
moments = utils.compute_moments(dl_tr.dataset)
print('Moments found: {}'.format(moments.tolist()))

# Set normalization using statistics
dl_tr.dataset.transform.transforms[-1] = transforms.Normalize(
    mean=moments[0].tolist(), std=moments[1].tolist(),
)
dl_ev.dataset.transform.transforms[-1] = transforms.Normalize(
    mean=moments[0].tolist(), std=moments[1].tolist(),
)
print('Transformations (train) replaced:\n{}'.format(dl_tr.dataset.transform))
print('Transformations (eval) replaced:\n{}'.format(dl_ev.dataset.transform))

try:
    filepath = 'stats.json'
    with open(filepath, 'w') as path:
        json.dump(
            {
                'meanRGB': moments[0].tolist(),
                'stdRGB': moments[1].tolist(),
            },
            path,
            indent=4,
        )
except:
    pass

model = net.bn_inception(pretrained=True)
net.embed(model, sz_embedding=args.sz_embedding)
model = model.cuda()

criterion = config['criterion']['type'](
    nb_classes=dl_tr.dataset.nb_classes(),
    sz_embed=args.sz_embedding,
    **config['criterion']['args']
).cuda()

opt = config['opt']['type'](
    [
        {  # inception parameters, excluding embedding layer
            **{'params': list(
                set(
                    model.parameters()
                ).difference(
                    set(model.embedding_layer.parameters())
                )
            )},
            **config['opt']['args']['backbone']
        },
        {  # embedding parameters
            **{'params': model.embedding_layer.parameters()},
            **config['opt']['args']['embedding']
        },
        {  # proxy nca parameters
            **{'params': criterion.parameters()},
            **config['opt']['args']['proxynca']
        }
    ],
    **config['opt']['args']['base']
)

scheduler = config['lr_scheduler']['type'](
    opt, **config['lr_scheduler']['args']
)

imp.reload(logging)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
        logging.StreamHandler()
    ]
)

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()
logging.info("**Evaluating initial model...**")
with torch.no_grad():
    utils.evaluate(model, dl_ev)

it = 0

for e in range(0, args.nb_epochs):
    scheduler.step(e)
    time_per_epoch_1 = time.time()
    losses_per_epoch = []

    for x, y, _ in dl_tr:
        it += 1
        opt.zero_grad()
        m = model(x.cuda())
        loss = criterion(m, y.cuda())
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch[-20:]))
    print('it: {}'.format(it))
    print(opt)
    logging.info(
        "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            time_per_epoch_2 - time_per_epoch_1
        )
    )
    with torch.no_grad():
        logging.info("**Evaluating...**")
        scores.append(utils.evaluate(model, dl_ev))
    model.losses = losses
    model.current_epoch = e

    # Save model every 5 epochs
    if e % 5 == 0:
        save_model(suffix='{:03}'.format(e))

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))

os.makedirs('model', exist_ok=True)
logging.info("Saving final model ...")
try_n_times(fn=save_model, ntimes=5, suffix='final')
logging.info("Saved")
