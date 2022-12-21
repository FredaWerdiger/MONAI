from inference import *
from monai.transforms import SplitDimd
def main(root_dir, ctp_df, model_path, out_tag, ddp=False):

    # test on external data
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            SplitDimd(keys="image", dim=0, keepdim=True,
                      output_postfixes=['b1000', 'adc']),
            Resized(keys=["image", "image_b1000", "image_adc"],
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=(128, 128, 128)),
            NormalizeIntensityd(keys="image_b1000", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image_b1000", "label"]),
            # SaveImaged(keys="image", output_dir=root_dir + "out", output_postfix="transform", resample=False)
        ]
    )

    test_files = make_dict(root_dir, 'test')
    test_ds = Dataset(
        data=test_files, transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys=["pred", "label"]),
        EnsureChannelFirstd(keys="label"),
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
        SaveImaged(keys="pred",
                   meta_keys="pred_meta_dict",
                   output_dir=root_dir + "out_" + out_tag + '/pred',
                   output_postfix="pred", resample=False,
                   separate_folder=False),
    ])

    if not os.path.exists(root_dir + "out_" + out_tag + '/pred'):
        os.makedirs(root_dir + "out_" + out_tag + '/pred')

    # removing sync on step as we are running on master node
    dice_metric = Dice(ignore_index=0)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    loader = LoadImage(image_only=False)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    if ddp:
        model.load_state_dict(ddp_state_dict(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    results = pd.DataFrame(columns=['id', 'dice', 'size', 'px_x', 'px_y', 'px_z', 'size_ml'])
    results['id'] = ['test_' + str(item).zfill(3) for item in range(1, len(test_loader) + 1)]

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image_b1000"].to(device)

            roi_size = (64, 64, 64)
            sw_batch_size = 2
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_output, test_label, test_image = from_engine(["pred", "label", "image_b1000"])(test_data)

            a = dice_metric(y_pred=test_output, y=test_label)

            dice_score = round(a.item(), 4)
            print(f"Dice score for image: {dice_score:.4f}")

            # get original image, and normalize it so we can see the normalized image
            # this is both channels
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            volx, voly, volz = original_image[1]['pixdim'][1:4] # meta data
            pixel_vol = volx * voly * volz

            original_image = original_image[0] # image data
            original_adc = original_image[:, :, :, 1]
            original_image = original_image[:, :, :, 0]
            ground_truth = test_label[0][1].detach().numpy()
            prediction = test_output[0][1].detach().numpy()
            transformed_image = test_inputs[0][0].detach().cpu().numpy()
            size = ground_truth.sum()
            size_ml = size * pixel_vol / 1000
            name = "test_" + os.path.basename(
                test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]
            save_loc = root_dir + "out_" + out_tag + "/images/" + name + "_"

            if not os.path.exists(root_dir + "out_" + out_tag + "/images/"):
                os.makedirs(root_dir + "out_" + out_tag + "/images/")

            create_paper_img(
                original_image,
                ground_truth,
                prediction,
                save_loc + "paper.png",
                define_dvalues(original_image),
                'png',
                dpi=300
            )
            create_mr_img(
                original_image,
                save_loc + "dwi.png",
                define_dvalues(original_image),
                'png',
                dpi=300)
            create_adc_img(
                original_adc,
                save_loc + "adc.png",
                define_dvalues(original_image),
                'png',
                dpi=300)

            [create_mrlesion_img(
                original_image,
                im,
                save_loc + name + '.png',
                define_dvalues(original_image),
                'png',
                dpi=300) for im, name in zip([prediction, ground_truth], ["pred", "truth"])]

            create_mr_big_img(transformed_image,
                              save_loc + "dwi_tran.png",
                              define_dvalues_big(transformed_image),
                              'png',
                              dpi=300)

            # uncomment below to visualise results.
            # plt.figure("check", (24, 6))
            # plt.subplot(1, 4, 1)
            # plt.imshow(original_image[:, :, 12], cmap="gray")
            # plt.title(f"image {name}")
            # plt.subplot(1, 4, 2)
            # plt.imshow(test_image[0].detach().cpu()[0, :, :, 12], cmap="gray")
            # plt.title(f"transformed image {name}")
            # plt.subplot(1, 4, 3)
            # plt.imshow(test_label[0].detach().cpu()[:, :, 12])
            # plt.title(f"label {name}")
            # plt.subplot(1, 4, 4)
            # plt.imshow(test_output[0].detach().cpu()[1, :, :, 12])
            # plt.title(f"Dice score {dice_score}")
            # plt.show()

            results.loc[results.id == name, 'size'] = size
            results.loc[results.id == name, 'size_ml'] = size_ml
            results.loc[results.id == name, 'px_x'] = volx
            results.loc[results.id == name, 'px_y'] = voly
            results.loc[results.id == name, 'px_z'] = volz
            results.loc[results.id == name, 'dice'] = dice_score

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print(f"Mean dice on test set: {metric:.4f}")

    results['mean_dice'] = metric

    from sklearn.cluster import k_means
    kmeans_labels = k_means(
        np.reshape(np.asarray(results['size'].to_list()), (-1,1)),
        n_clusters=2,
        random_state=0)[1]
    kmeans_labels = ["small-medium" if label==0 else "medium-large" for label in kmeans_labels]
    results['size_label']=kmeans_labels
    results_join = results.join(
        ctp_df[~ctp_df.index.duplicated(keep='first')],
        on='id',
        how='left')
    print(results)
    results_join.to_csv(root_dir + 'out_' + out_tag + '/results.csv', index=False)

    for sub in results_join['id']:
        create_overviewhtml(sub, results_join, root_dir + 'out_' + out_tag + '/')

if __name__ == '__main__':
    HOMEDIR = os.path.expanduser("~/")
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            '/home/unimelb.edu.au/fwerdiger/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
            index_col='dl_id'
        )
    elif os.path.exists('/media/mbcneuro'):
        directory = '/media/mbcneuro/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            '/home/mbcneuro/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
            index_col='dl_id'
        )
    elif os.path.exists('D:'):
        directory = 'D:/ctp_project_data/DWI_Training_Data/'
        ctp_df = pd.read_csv(
            'C:/Users/fwerdiger/PycharmProjects/study_design/study_lists/dwi_inspire_dl.csv',
            index_col='dl_id')

    model_path = directory + 'out_unet_recursive_from_scratch_dwi_only/best_metric_model600.pth'
    out_tag = 'unet_recursive_from_scratch_dwi_only'
    main(directory, ctp_df, model_path, out_tag)