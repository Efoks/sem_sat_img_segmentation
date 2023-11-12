import torch
import torchvision
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
import src.config as config
import src.utils as utils
import src.data_handling as dh
from src.models import prepare_deeplabv3_resnet101

torch.cuda.empty_cache()

config.print_model_config(config.resnet101)

def launch_model():

    # num_epochs = config.resnet101['num_epochs']
    num_epochs = 3
    batch_size = config.resnet101['batch_size']

    supervised_loader_train, supervised_loader_val, unsupervised_loader_train, unsupervised_loader_val = dh.create_data_loaders(batch_size, 0.5, perform_stratiication= False)

    model = prepare_deeplabv3_resnet101()

    writer = SummaryWriter(config.DEEPLABV3_RESNET101_LOG_DIR)

    model_wrapper = utils.wrap_the_model(model)
    sample_data = next(iter(supervised_loader_train))[0].to('cuda')
    image_grid = torchvision.utils.make_grid(sample_data)
    writer.add_image('tinyMiniFrance Image', image_grid)
    writer.add_graph(model_wrapper, sample_data)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(trainable_parameters)
    criterion = nn.BCEWithLogitsLoss()

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
                total_accuracy += total_accuracy

        avg_loss = total_loss / len(supervised_loader_val)
        avg_accuracy = total_accuracy / len(supervised_loader_val)
        writer.add_scalar('Validation Loss', avg_loss, epoch)
        writer.add_scalar('Accuracy', avg_accuracy, epoch)

    writer.close()

if __name__ == "__main__":
   launch_model()