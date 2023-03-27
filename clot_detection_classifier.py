import os
import sys

import pandas as pd

sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
import glob
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import SimpleITK as sitk
import numpy as np
import random
from sklearn.metrics import classification_report, roc_curve, auc
from monai.data import Dataset
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    EnsureType,
    LoadImage,
    NormalizeIntensity,
    RandAffine,
    RandFlip,
    RandRotate,
    RandZoom,
    RandAffine,
    Resize,
    ScaleIntensity,
    ThresholdIntensity
)
from monai.utils import set_determinism
from torch.nn import DataParallel as DDP
import time
from sklearn.model_selection import train_test_split

print_config()

HOME = os.path.expanduser('~/')
image_size = [128]

if os.path.exists(HOME + 'mediaflux'):
    mediaflux = os.path.join(HOME, 'mediaflux')
    directory = os.path.join(mediaflux, 'CTA', 'CODEC-IV', 'INSPIRE - do not share')
elif os.path.exists('Z:/data_freda/'):
    directory = os.path.join('Z:', 'CTA', 'CODEC-IV', 'INSPIRE - do not share')
elif os.path.exists('/data/gpfs/projects/punim1086/clot_detection'):
    directory = '/data/gpfs/projects/punim1086/clot_detection/'


out_directory = os.path.join(directory, 'batch16')
image_files_list = glob.glob(os.path.join(directory, 'codec_skullstrip/*'))
image_files_list.sort()

subjects_1 = [file.split('skullstrip/')[1].split('_cta')[0] for file in image_files_list]

segmentations = [file for file in glob.glob(os.path.join(directory, 'segmentations_all/*'))
                 if any(name in file for name in subjects_1)]
segmentations.sort()

subjects_2 = [file.split('segmentations_all/')[1].split('_seg')[0] for file in segmentations]

assert subjects_1 == subjects_2


image_class = []
for file in segmentations:
    im = sitk.ReadImage(file)
    label = sitk.LabelShapeStatisticsImageFilter()
    label.Execute(im)
    labels = label.GetLabels()
    if len(labels) > 0:
        image_class.extend([1])
    else:
        image_class.extend([0])

print(f"Number of patients with a visible clot: {image_class.count(1)}")
print(f"Number of patients with no visible clot: {image_class.count(0)}")

# add more no occlusion cases
class_names = [0, 1]
num_total = len(image_files_list)

image_files_new = glob.glob(os.path.join(directory, 'no_occlusion_checked/*'))
subjects_new = [file.split('no_occlusion_checked/')[1].split('_cta')[0] for file in image_files_new]

for subject in subjects_new:
    image_file = [file for file in image_files_new if subject in file][0]
    image_class.append(0)
    image_files_list.append(image_file)

print(f"Number of patients with a visible clot: {image_class.count(1)}")
print(f"Number of patients with no visible clot: {image_class.count(0)}")

# Randomly pick image to display
#
# plt.subplots(2, 2, figsize=(8, 8))
# for i, k in enumerate(np.random.randint(num_total, size=4)):
#     im = sitk.ReadImage(image_files_list[k])
#     size = im.GetSize()
#     slice = random.randint(10, size[2] - 10)
#     arr = sitk.GetArrayFromImage(im)
#     plt.subplot(2, 2, i + 1)
#     plt.xlabel(image_class[k])
#     plt.imshow(arr[slice], cmap="gray", vmin=0, vmax=255)
# plt.tight_layout()
# plt.show()
# plt.close()

train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

length = len(image_files_list)
indices = np.arange(length)
random_state = 42

# make a dataframe with data

df = pd.DataFrame( columns=['subject', 'index', 'class'])
df['index'] = indices
df['class'] = image_class
df['subject'] = subjects_1 + subjects_new

num_train = int(np.ceil(train_frac * length))
num_validation = int(np.ceil(val_frac * length))
num_test = length - (num_train + num_validation)

train_indices, test_indices = train_test_split(indices,
                                     train_size=num_train,
                                     test_size=num_test+num_validation,
                                     random_state=random_state,
                                     shuffle=True,
                                     stratify=image_class)

test_labels = df[df.apply(lambda x: x['index'] in test_indices, axis=1)]['class'].to_list()
test_indices = df[df.apply(lambda x: x['index'] in test_indices, axis=1)]['index'].to_list()


val_indices, test_indices = train_test_split(test_indices,
                                     train_size=num_validation,
                                     test_size=num_test,
                                     random_state=random_state,
                                     shuffle=True,
                                     stratify=test_labels)


train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")


class CodecDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(mode='trilinear', align_corners=True, spatial_size=image_size * 3),
        ThresholdIntensity(threshold=80, above=False),
        ThresholdIntensity(threshold=0, above=True),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        RandAffine(prob=0.5, translate_range=10),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandFlip(spatial_axis=2, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)

val_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(mode='trilinear', align_corners=True, spatial_size=image_size * 3),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        EnsureType(),
    ]
)


y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=2)])

batch_size = 16
train_ds = CodecDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = CodecDataset(val_x, val_y, val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size)

test_ds = CodecDataset(test_x, test_y, val_transforms)
test_loader = DataLoader(test_ds, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
# model = DDP(model)
model = model.to(device)
weights = [1, 1]
class_weights = torch.FloatTensor(weights).to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
learning_rate = 1e-4
weight_decay = 1e-4
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
max_epochs = 200

val_interval = 1
auc_metric = ROCAUCMetric()

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

start = time.time()

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(out_directory,
                                                            "best_metric_model"
                                                            + model._get_name() + ".pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

end = time.time()
time_taken = end - start
print(f"Time taken: {round(time_taken, 0)} seconds")
time_taken_hours = time_taken / 3600
time_taken_mins = np.ceil((time_taken / 3600 - int(time_taken / 3600)) * 60)
time_taken_hours = int(time_taken_hours)


plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val AUC")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.savefig(os.path.join(directory, "loss_plot_" + model._get_name() + ".png"),
            bbox_inches='tight', dpi=300, format='png')
plt.close()

model.load_state_dict(torch.load(os.path.join(out_directory,
                                              "best_metric_model"
                                              + model._get_name() + ".pth")))
model.eval()
y_true_array = []
y_pred_array = []
with torch.no_grad():
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = model(test_images).argmax(dim=1)

        for i in range(len(pred)):
            y_true_array.append(test_labels[i].item())
            y_pred_array.append(pred[i].item())

fpr, tpr, thresholds = roc_curve(y_true_array, y_pred_array, pos_label=1)

roc = pd.DataFrame(columns=['fpr', 'tpr', 'thresholds'])
roc['fpr'] = fpr
roc['tpr'] = tpr
roc['thresholds'] = thresholds

roc.to_csv(out_directory + '/roc_values_' +  model._get_name() + '.csv', index=False)

auc = auc(fpr, tpr)

print(f"AUC on test set: {auc}")
print(classification_report(y_true_array, y_pred_array, target_names=['0', '1'], digits=4))

model_name = model._get_name()
loss_name = loss_function._get_name()

with open(out_directory + '/model_info_' + str(
        max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_batch' + str(batch_size) + '.txt', 'w') as myfile:
    myfile.write(f'Train dataset size: {len(train_x)}\n')
    myfile.write(f'Validation dataset size: {len(val_x)}\n')
    myfile.write(f'Testing dataset size: {len(test_x)}\n')
    myfile.write(f"Number of patients with a visible clot: {image_class.count(1)}\n")
    myfile.write(f"Number of patients with no visible clot: {image_class.count(0)}\n")
    myfile.write(f'Model: {model_name}\n')
    myfile.write(f'Loss function: {loss_name}\n')
    myfile.write(f'Number of epochs: {max_epochs}\n')
    myfile.write(f'Learning rate: {learning_rate}\n')
    myfile.write(f'Weight decay: {weight_decay}\n')
    myfile.write(f'Batch size: {batch_size}\n')
    myfile.write(f'Image size: {image_size}\n')
    myfile.write(f'Validation interval: {val_interval}\n')
    myfile.write(f"Best metric: {best_metric:.4f}\n")
    myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
    myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")

