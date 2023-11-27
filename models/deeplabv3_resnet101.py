import os
import torch
import torchvision
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics as metrics
import src.config as config
import src.utils as utils
import src.data_handling as dh
from src.models import prepare_deeplabv3_resnet101

torch.cuda.empty_cache()

config.print_model_config(config.resnet101)

def launch_model():

    experiment = '_exp_1_tversky_loss'

    num_epochs = config.resnet101['num_epochs']
    # num_epochs = 3
    batch_size = config.resnet101['batch_size']
    jaccard_metric = metrics.JaccardIndex(num_classes=config.NUM_CLASSES, task='multiclass').to('cuda')
    dice_metric = metrics.Dice(num_classes=config.NUM_CLASSES).to('cuda')

    supervised_loader_train, supervised_loader_val, unsupervised_loader_train, unsupervised_loader_val = dh.create_data_loaders_deeplab(batch_size, perform_stratiication= False)

    model = prepare_deeplabv3_resnet101()

    experiment_dir = utils.create_experiment_dir(exp_name=experiment, model_name='deeplabv3_' + config.resnet101['backbone'])

    writer = SummaryWriter(experiment_dir)

    model_wrapper = utils.wrap_the_model(model)
    sample_data = next(iter(supervised_loader_train))[0].to('cuda')
    image_grid = torchvision.utils.make_grid(sample_data)
    writer.add_image('tinyMiniFrance Image', image_grid)
    writer.add_graph(model_wrapper, sample_data)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(trainable_parameters)
    criterion = utils.TverskyLoss()

    for epoch in range(num_epochs):
        print(f'Started {epoch+1} epoch')

        model.train()
        for batch_idx, (image, mask) in enumerate(supervised_loader_train):
            image, mask = image.to('cuda'), mask.to('cuda')
            optimizer.zero_grad()
            output = model(image)
            output_prediction = output['out']

            loss = criterion(output_prediction, mask)
            writer.add_scalar('Training Loss', loss.item(), epoch * len(supervised_loader_train) + batch_idx)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for batch_idx, (image, mask) in enumerate(supervised_loader_val):
                image, mask = image.to('cuda'), mask.to('cuda')
                output = model(image)
                output_prediction = output['out']
                loss = criterion(output_prediction, mask)
                total_loss += loss.item()

                accuracy = utils.calculate_accuracy(output_prediction, mask)
                total_accuracy += accuracy

                predicted, true_mask = utils.prepare_output_for_calculation(output_prediction, mask)
                jaccard_metric.update(predicted, true_mask)
                dice_metric.update(predicted, true_mask)

        avg_loss = total_loss / len(supervised_loader_val)
        avg_accuracy = total_accuracy / len(supervised_loader_val)
        writer.add_scalar('Average Validation Loss', avg_loss, epoch)
        writer.add_scalar('Average Accuracy in Validation', avg_accuracy, epoch)

    jaccard_score = jaccard_metric.compute()
    dice_score = dice_metric.compute()

    torch.save(model.state_dict(), os.path.join(config.DEEPLABV3_RESNET101_SAVE_DIR,
                                                f'deeplabv3_{config.resnet101["backbone"]}_{experiment}_state_dict.pth'))

    score_table = f"""
        | Metric | Score  | 
        |----------|-----------|
        | Jaccard Score    | {jaccard_score:.2f} |
        | Dice Score    | {dice_score:.2f} |
    """
    score_table = '\n'.join(l.strip() for l in score_table.splitlines())
    writer.add_text("Score Table", score_table, 0)

    writer.close()

if __name__ == "__main__":
   launch_model()