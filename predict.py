import numpy as np
import torch
from sklearn.utils.extmath import softmax

import evaluation
import dataset
import net
from dataset.crops import Crops
from utils import predict_batchwise

transformations = {
    "rgb_to_bgr": False,
    "rgb_to_hsv": True,
    "intensity_scale": [[0, 1], [0, 255]],
    "mean": [149.2224, 66.9379, 140.6337],
    "std": [55.72, 71.4796, 70.2449],
    "is_train": False,
}


def load_model(path):
    model = net.bn_inception(pretrained=True)
    net.embed(model, sz_embedding=64)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def emulate_input_data():
    """ Emulates input data """
    base_dataset = dataset.load(
        name='aic19',
        root='/home/jon/repos/mcv/m6/proxy-nca/data/train/S01/c001',
        classes=range(0, 77),
        transform=None,
    )

    crops = []
    ys = []
    # for idx in range(len(base_dataset)):
    for idx in range(6):
        im, label, _ = base_dataset[idx]
        crops.append(im)
        ys.append(label)

        # Visualise untransformed crops
        # im.show()

    return crops[0:2], crops[2:6]


def get_dataloader(samples):
    ds = Crops(
        imgs=samples,
        transform=dataset.utils.make_transform(**transformations),
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # DEBUG test dataset
    # for idx in range(1):
    #     im, label, _ = ds[idx]
    #
    #     # Visualise transformed crops
    #     sample_arr = np.transpose(im.numpy().astype(np.uint8), (1, 2, 0))
    #     sample = PIL.Image.fromarray(sample_arr)
    #     sample.show()
    return dl


def do_it(samples0, samples1, verbose=False):
    """ Maps elements in `samples0` from `samples1`

    Args:
        samples0: list of PIL images
        samples1: list of PIL images
        verbose: set True to print some metrics

    Returns:
        array of size `samples0` with indices on `samples1`, confidence
    """
    model = load_model(path="model/{}".format('test-aic19_015.pt'))

    dl0_test = get_dataloader(samples0)
    dl1_test = get_dataloader(samples1)

    # Embeddings, Targets, Predictions (k=1 => most probable class)
    # print(X0, T0, Y0, Y0.shape)
    X0, T0, *_ = predict_batchwise(model, dl0_test)
    Y0 = evaluation.assign_by_euclidian_at_k(X0, T0, 1)
    Y0 = torch.from_numpy(Y0)

    X1, T1, *_ = predict_batchwise(model, dl1_test)
    Y1 = evaluation.assign_by_euclidian_at_k(X1, T1, 1)
    Y1 = torch.from_numpy(Y1)

    distances_between_sets = evaluation.get_distances(X0, X1)
    prob_matrix = softmax(-distances_between_sets)
    matches = np.argmax(prob_matrix, axis=1)
    confidence = np.max(prob_matrix, axis=1)

    if verbose:
        print('Distance matrix: \n{}'.format(distances_between_sets))
        print('Prob matrix: \n{}:'.format(prob_matrix))
        print('Distance to most prob: \n{}'.format(matches))
        print('Confidence to most prob: \n{}'.format(confidence))

    return matches, confidence


if __name__ == '__main__':
    frame0_samples, frame1_samples = emulate_input_data()
    matches, confidences = do_it(frame0_samples, frame1_samples, verbose=False)