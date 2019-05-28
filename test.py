import seaborn as sns
from utils.Scorekeeper import Scorekeeper
import pandas as pd

def test(test_data_loader, score_keeper, plot=True, channels=3):
    """
    Testing of network
    :param test_data_loader: Data to test
    :param plot: If to plot predictions
    :return:
    """
    def Smoothen(batch):
        """
        If you want to smoothen (Gaussian) the output images
        NOT USED
        """
        tran = transforms.ToTensor()
        for i in range(batch.size()[0]):
            if i == 0:
                inb = batch[i, :, :, :].numpy().transpose((1, 2, 0))
                inb = ndimage.gaussian_filter(inb, sigma=(1, 1, 0))
                out = tran(inb).unsqueeze_(0)
            else:
                inb = batch[i, :, :, :].numpy().transpose((1, 2, 0))
                inb = ndimage.gaussian_filter(inb, sigma=(1, 1, 0))
                inb = tran(inb).unsqueeze_(0)
                out = torch.cat((out, inb), dim=0)
        return out

    def initial_input(No_more_Target):
        output = model(ImageSeries.to(device))
        try:
            target = OriginalSeries[:, (t0 + cnt + input_frames) * channels:(t0 + cnt + input_frames + 1) * channels, :, :].to(device)
        except:
            No_more_Target = True
            target = None
        return output, target, No_more_Target

    def new_input(output, target, No_more_Target):
        output = torch.cat((output, model(ImageSeries, mode="new_input")), dim=1)
        try:
            target = torch.cat(
                (target, OriginalSeries[:, (t0 + cnt + input_frames) * channels:(t0 + cnt + input_frames + 1) * channels, :, :].to(device)
                 ), dim=1)
        except:
            No_more_Target = True
        return output, target, No_more_Target

    def consequent_propagation(output, target, No_more_Target):
        if n < (output_frames - refeed_offset):
            output = torch.cat((output, model(torch.Tensor([0]), mode="internal")), dim=1)
            try:
                target = torch.cat(
                    (target, OriginalSeries[:, (t0 + cnt + input_frames) * channels:(t0 + cnt + input_frames + 1) * channels, :, :].to(device)
                ), dim=1)
            except:
                No_more_Target = True
        return output, target, No_more_Target

    def plot_predictions():
        if (total == 0) & (n == 0) & (run == 0):
            for imag in range(int(ImageSeries.shape[1] / channels)):
                fig = plt.figure().add_axes()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                sns.set_context("talk")
                imshow(ImageSeries[selected_batch, imag * channels:(imag + 1) * channels, :, :], title="Input %01d" % imag, obj=fig)
                figure_save(maindir1 + "Input %02d" % imag)
        if (total == 0) & (n < (output_frames - refeed_offset)):
            predicted = output[selected_batch, -channels:, :, :].cpu()
            des_target = target[selected_batch, -channels:, :, :].cpu()
            fig = plt.figure()
            sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
            sns.set_context("talk")
            pred = fig.add_subplot(1, 2, 1)
            imshow(predicted, title="Predicted %02d" % cnt, smoothen=True, obj=pred)
            tar = fig.add_subplot(1, 2, 2)
            imshow(des_target, title="Target %02d" % target_cnt, obj=tar)
            figure_save(maindir1 + "Prediction %02d" % cnt)
            plt.show() if plot else plt.close()

    def plot_cutthrough(frequently_plot=5, direction="Horizontal", location=None):
        def cutthrough(img1, img2,  hue1, hue2):
            intensity = []
            location = []
            hue = []
            if "orizontal" in direction:
                for i in range(np.shape(img1)[1]):
                    intensity.append(img1[stdmax[0], i, 0])
                    location.append(i)
                    hue.append(hue1)
                for i in range(np.shape(img2)[1]):
                    intensity.append(img2[stdmax[0], i, 0])
                    location.append(i)
                    hue.append(hue2)
            elif "ertical" in direction:
                for i in range(np.shape(img1)[0]):
                    intensity.append(img1[i, stdmax[0], 0])
                    location.append(i)
                    hue.append(hue1)
                for i in range(np.shape(img2)[0]):
                    intensity.append(img2[i, stdmax[0], 0])
                    location.append(i)
                    hue.append(hue2)

            data_dict = {"Intensity": intensity, "Pixel Location": location, "Image": hue}
            #g = sns.FacetGrid(pd.DataFrame.from_dict(data_dict), col="Image")
            #g.map(sns.lineplot, "Pixel Location", "Intensity")
            sns.lineplot(x="Pixel Location", y="Intensity", hue="Image",
                         data=pd.DataFrame.from_dict(data_dict), ax=profile)
            profile.set_title("Intensity Profile")

        if total == 0:
            if ((cnt + 1) % frequently_plot) == 0 or (cnt == 0):
                predicted = output[selected_batch, -channels:, :, :].cpu()
                des_target = target[selected_batch, -channels:, :, :].cpu()
                fig = plt.figure()
                sns.set(style="white")  # darkgrid, whitegrid, dark, white, and ticks
                with sns.axes_style("white"):
                    pre = fig.add_subplot(2, 2, 1)
                    tar = fig.add_subplot(2, 2, 2)
                with sns.axes_style("darkgrid"):  # darkgrid, whitegrid, dark, white, and ticks
                    profile = fig.add_subplot(2, 2, (3, 4))

                predicted = imshow(predicted, title="Predicted %02d" % cnt, return_np=True, obj=pre)
                des_target = imshow(des_target, title="Target %02d" % target_cnt, return_np=True, obj=tar)
                if not location:
                    if "orizontal" in direction:
                        std = np.std(des_target, axis=1)
                    elif "ertical" in direction:
                        std = np.std(des_target, axis=0)
                    stdmax, _ = np.where(std.max() == std)
                else:
                    stdmax = location

                if "orizontal" in direction:
                    pre.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
                    tar.plot([0, np.shape(std)[0]], [stdmax[0], stdmax[0]], color="yellow")
                elif "ertical" in direction:
                    pre.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")
                    tar.plot([stdmax[0], stdmax[0]], [0, np.shape(std)[0]], color="yellow")

                cutthrough(predicted, des_target, "Predicted", "Target")
                figure_save(maindir1 + "Cut-through %02d" % cnt, obj=fig)
                plt.show() if plot else plt.close()

    def add_score():
        if (not No_more_Target) & (n < (output_frames - refeed_offset)):
            for ba in range(output.size()[0]):
                score_keeper.add(output[ba, -channels:, :, :].cpu(), target[ba, -channels:, :, :].cpu(), cnt,
                                 "pHash", "SSIM", "Own", "RMSE")

    def introduce(prev_data):
        """
        If you want to introduce new droplets live during simulation
        NOT USED
        """
        def find_mean(input_img):
            for k in range(int(input_img.size()[0])):
                mean, number = np.unique(input_img[k:k + 1, :, :], return_counts=True)
                mean = np.full(np.shape(input_img[k:k + 1, :, :]), mean[np.argmax(number)])
                mean = torch.Tensor([mean])
                if k == 0:
                    matrix = mean
                else:
                    matrix = torch.cat((matrix, mean), dim=1)
            return matrix.squeeze_(0)

        prev_data = prev_data.cpu()
        data = My_Test[0]["image"][t0 * channels: (t0 + input_frames) * channels, :, :]
        for i in range(int(data.size()[0] / channels)):
            means = find_mean(data[i * channels:(i + 1) * channels, :, :])
            prev_data[selected_batch, i * channels:(i + 1) * channels, :, :] += data[i * channels:(i + 1) * channels, :, :] - means
        return prev_data


    model.eval()
    correct = total = 0
    t0 = 15 # Can be 0
    refeed_offset = 0
    selected_batch = random.randint(0, 15)
    if (output_frames - refeed_offset) < input_frames:
        refeed_offset = output_frames - input_frames
    for batch_num, batch in enumerate(test_data_loader):
        OriginalSeries = batch["image"]
        ImageSeries = OriginalSeries[:, t0 * channels:(t0 + input_frames) * channels, :, :]
        #ImageSeries = introduce(ImageSeries)
        model.reset_hidden(ImageSeries.size()[0])
        No_more_Target = False
        cnt = target_cnt = 0
        for run in range(int(m.ceil((100 - (t0 + input_frames + 1)) / (output_frames - refeed_offset)))):
            if run != 0:
                if (refeed_offset == 0) or ((output_frames - refeed_offset) <= input_frames):
                    ImageSeries = output[:, -input_frames * channels:, :, :]
                else:
                    ImageSeries = output[:, -(input_frames + refeed_offset) * channels:-refeed_offset * channels, :, :]
                cnt -= refeed_offset
            for n in range(output_frames):
                if n == 0:
                    if run == 0:
                        output, target, No_more_Target = initial_input(No_more_Target)
                    else:
                        output, target, No_more_Target = new_input(output, target, No_more_Target)
                else:
                    output, target, No_more_Target = consequent_propagation(output, target, No_more_Target)
                    # output & target size is [batches, 3 * (n + 1), 128, 128]

                add_score()
                plot_predictions()
                plot_cutthrough()
                cnt += 1
                if not No_more_Target:
                    target_cnt = copy.copy(cnt)

        total += target.size()[0]
        logging.info(batch_num + 1, "out of", len(test_data_loader))
    logging.info("Correct: {}\tPercentile: {:.0f}%".format(correct, 100 * correct / total))
    score_keeper.plot()

score_keeper = Scorekeeper()
test(Test_Data, score_keeper, plot=True, channels=channels)