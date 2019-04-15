import PIL
import numpy as np
import torch
from dataset.crops import Crops
from sklearn.utils.extmath import softmax
import evaluation
import net
import dataset
from utils import evaluate, predict_batchwise

model = net.bn_inception(pretrained=True)
net.embed(model, sz_embedding=64)

model_path = "model/{}".format('test-aic19_015.pt')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

transformations = {
    "rgb_to_bgr": False,
    "rgb_to_hsv": True,
    "intensity_scale": [[0, 1], [0, 255]],
    "mean": [149.2224, 66.9379, 140.6337],
    "std": [55.72, 71.4796, 70.2449],
    "is_train": False,
}


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


frame0_samples, frame1_samples = emulate_input_data()

ds0_test = Crops(
    imgs=frame0_samples,
    transform=dataset.utils.make_transform(**transformations),
)

ds1_test = Crops(
    imgs=frame1_samples,
    transform=dataset.utils.make_transform(**transformations),
)

# DEBUG test dataset
# for idx in range(1):
#     im, label, _ = ds0_test[idx]
#
#     # Visualise transformed crops
#     sample_arr = np.transpose(im.numpy().astype(np.uint8), (1, 2, 0))
#     sample = PIL.Image.fromarray(sample_arr)
#     sample.show()

dl0_test = torch.utils.data.DataLoader(
    ds0_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)
dl1_test = torch.utils.data.DataLoader(
    ds1_test,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)

X0, T0, *_ = predict_batchwise(model, dl0_test)
Y0 = evaluation.assign_by_euclidian_at_k(X0, T0, 1)
Y0 = torch.from_numpy(Y0)

# Embeddings, Targets, Predictions (k=1 => most probable class)
print(X0, T0, Y0, Y0.shape)

X1, T1, *_ = predict_batchwise(model, dl1_test)
Y1 = evaluation.assign_by_euclidian_at_k(X1, T1, 1)
Y1 = torch.from_numpy(Y1)

# NOTE implement a function to map two lists of crops for similarity
distances_between_sets = evaluation.get_distances(X0, X0)
print('Distance matrix: \n{}'.format(distances_between_sets))

prob_matrix = 1.0 - softmax(distances_between_sets)
print('Prob matrix: \n{}:'.format(prob_matrix))

most_probable = np.argmax(prob_matrix, axis=1)
print('Distance to most prob: \n{}'.format(most_probable))

confidence = np.max(prob_matrix, axis=1)
print('Confidence to most prob: \n{}'.format(confidence))
