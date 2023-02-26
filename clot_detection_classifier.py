import os
import sys
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
from sklearn.metrics import classification_report
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
    Resize,
    ScaleIntensity,
    ThresholdIntensity
)
from monai.utils import set_determinism

print_config()

HOME = os.path.expanduser('~/')
image_size = [128]

if os.path.exists(HOME + 'mediaflux'):
    mediaflux = os.path.join(HOME, 'mediaflux')
    directory = os.path.join(mediaflux, 'CTA', 'CODEC-IV', 'INSPIRE - do not share')
elif os.path.exists('Z:/data_freda/'):
    directory = os.path.join('Z:', 'CTA', 'CODEC-IV', 'INSPIRE - do not share')
elif os.path.exists('/data/gpfs/projects/punim1086/clot_detection'):
    directory = '/data/gpfs/projects/punim1086/clot_detection'


out_directory = os.path.join(directory, 'results')
image_files_list = glob.glob(os.path.join(directory, 'cta_all/*'))
image_files_list.sort()

subjects_1 = [file.split('cta_all/')[1].split('_cta')[0] for file in image_files_list]

# TODO: Replace with skull stripped image
#
# image_files_list = []
# for subject in subjects_1:
#     brain_im = glob.glob(mediaflux +
#                          '/' +
#                          'INSPIRE_database/' +
#                          subject +
#                          '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*mistar_brain.nii.gz')
#     if len(brain_im) > 1:
#         image_files_list.append(brain_im[0])
#     else:
#         brain_im = glob.glob(os.path.join(mediaflux,
#                                           'INSPIRE_database/'
#                                           + subject +
#                                           '/CT_baseline/CTP_baseline/mistar/Mean_baseline/*_brain.nii.gz'))[0]
#         image_files_list.append(brain_im)

segmentations = glob.glob(os.path.join(directory, 'segmentations_all/*'))
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

class_names = [0, 1]
num_total = len(image_files_list)

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

val_frac = 0.1
test_frac = 0.1
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

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


train_ds = CodecDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

val_ds = CodecDataset(val_x, val_y, val_transforms)
val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)

test_ds = CodecDataset(test_x, test_y, val_transforms)
test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
weights = [1, 0.5]
class_weights = torch.FloatTensor(weights).to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 4
val_interval = 1
auc_metric = ROCAUCMetric()

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

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
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

print(classification_report(y_true, y_pred, target_names=['0', '1'], digits=4))