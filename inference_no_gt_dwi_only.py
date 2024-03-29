
from inference import *
import sys
from monai.transforms import SplitDimd

def main(directory, model_file):

    test_files = BuildDataset(directory, "no_seg").no_seg_dict
    # running on scans from CN13 site
    test_files = [name for name in test_files if 'CN13' in name["image"]]

    # replace with path
    out_dir = directory + "no_seg/masks"
    trans_dir = directory + "no_seg/transforms"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(trans_dir):
        os.makedirs(trans_dir)

    # test on external data
    test_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['b1000', 'adc']),
            Resized(keys=["image", "image_b1000", "image_adc"],
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image_b1000", nonzero=True, channel_wise=True),
            EnsureTyped(keys="image_b1000"),
            SaveImaged(keys="image_b1000",
                       output_dir=trans_dir,
                       output_postfix="transform",
                       resample=False,
                       separate_folder=False)
        ]
    )

    test_ds = Dataset(
        data=test_files, transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image_b1000",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred",
                    argmax=True),
        SaveImaged(keys="pred",
                   meta_keys="pred_meta_dict",
                   output_dir=out_dir,
                   output_postfix="pred", resample=False, separate_folder=False),
    ])

    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    loader = LoadImage()

    model.load_state_dict(torch.load(model_file))
    model.eval()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image_b1000"].to(device)
            roi_size = (64, 64, 64)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            name = 'INSP_' + os.path.basename(
                test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[2]
            print(name)

            test_output, test_image = from_engine(["pred", "image"])(test_data)
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            # volx, voly, volz = original_image[1]['pixdim'][1:4]  # meta data

            original_image = original_image[0] # image data
            original_image = original_image[:, :, :, 0]
            prediction = test_output[0][0].detach().numpy()

            # save_loc = directory + "no_seg/pred_images/" + name + "_pred.png"

            # create_mrlesion_img(
            #     original_image,
            #     prediction,
            #     save_loc,
            #     define_dvalues(original_image),
            #     'png',
            #     dpi=300)



if __name__ == '__main__':
    directory = sys.argv[1]
    model_file = sys.argv[2]