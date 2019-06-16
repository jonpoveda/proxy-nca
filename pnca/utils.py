from __future__ import print_function
from __future__ import division

import evaluation
import numpy as np
import torch
import logging
import proxynca
import json

# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name='config.json'):
    config = json.load(open(config_name))

    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])

    eval_json(config)
    return config


def predict_batchwise(model, dataloader):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()
                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training)  # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]


def evaluate(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, *_ = predict_batchwise(model, dataloader)

    # calculate NMI with kmeans clustering
    nmi = evaluation.calc_normalized_mutual_information(
        T,
        evaluation.cluster_by_kmeans(
            X, nb_classes
        )
    )

    logging.info("NMI: {:.3f}".format(nmi * 100))

    # get predictions by assigning nearest 8 neighbors with euclidian
    Y = evaluation.assign_by_euclidian_at_k(X, T, 8)
    Y = torch.from_numpy(Y)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 2, 4, 8]:
        r_at_k = evaluation.calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return nmi, recall


def compute_moments(dataset):
    """ Computes mean and std of the dataset samples """
    moments1 = torch.tensor((), dtype=torch.float32)
    moments2 = torch.tensor((), dtype=torch.float32)

    print('Computing stadistics of the dataset', end=' ', flush=True)
    num_processed_samples = 0
    percent = len(dataset) // 100
    for image, _, _ in dataset:
        m1 = image.mean((1, 2))
        m2 = image.pow(2).mean((1, 2))

        moments1 = torch.cat((
            moments1,
            m1,
        ), dim=0)
        moments2 = torch.cat((
            moments2,
            m2,
        ), dim=0)

        if num_processed_samples % percent == 0:
            # Prints a dot on each percentage point processed
            print('.', end='', flush=True)
        num_processed_samples += 1

    # Stats by colour
    moment1 = mean = moments1.view((-1, 3)).mean(0)
    moment2 = var = moments2.view((-1, 3)).mean(0)
    std = torch.sqrt(torch.abs(var - mean.pow(2)))

    print('Done\n {}, {}'.format(mean, std), flush=True)
    return torch.stack((mean, std))
