import logging
import time
from pathlib import Path

import numpy as np
import torch
from SoccerNet.Evaluation.utils import AverageMeter
from sklearn.metrics import average_precision_score as avg_prec_score
from tqdm import tqdm


def trainer(loaders, model, optimizer, scheduler, criterion, writer, weights_dir: Path, max_epochs=1000,
            evaluation_frequency=20):
    logging.info("start training")

    best_loss = np.Inf
    running_train_loss, running_valid_loss = 0., 0.

    for epoch in range(max_epochs):
        best_model_path = weights_dir.joinpath("model.pth.tar")

        # train for one epoch
        training_loss = train(loaders.train, model, criterion, optimizer, epoch + 1)
        running_train_loss += training_loss

        # evaluate on validation set
        validation_loss = train(loaders.valid, model, criterion, optimizer, epoch + 1, True)
        running_valid_loss += validation_loss

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        if validation_loss < best_loss:
            torch.save(state, best_model_path)

        best_loss = min(validation_loss, best_loss)

        # Test the model on the validation set
        if (epoch + 1) % evaluation_frequency == 0:
            writer.add_scalar('training loss',
                              running_train_loss / evaluation_frequency,
                              (epoch + 1) * len(loaders.train))

            writer.add_scalar('validation loss',
                              running_valid_loss / evaluation_frequency,
                              (epoch + 1) * len(loaders.valid))
            validation_mAP = test(loaders.valid, model)
            logging.info(f"Validation mAP at epoch {epoch + 1} -> {validation_mAP}")

            running_train_loss, running_valid_loss = 0.0, 0.0

        # Reduce LR on Plateau after patience reached
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(validation_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr is not prev_lr and scheduler.num_bad_epochs == 0:
            logging.info("Plateau Reached!")

        if prev_lr < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            logging.info("Plateau Reached and no more reduction -> Exiting Loop")
            break
    return


def train(loader, model, criterion, optimizer, epoch, evaluate_=False):
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    description = '{mode} {epoch}: ' \
                  'Time {avg_time:.3f}s (it:{it_time:.3f}s) ' \
                  'Data:{avg_data_time:.3f}s (it:{it_data_time:.3f}s) ' \
                  'Loss {loss:.4e}'

    if evaluate_:
        model.eval()
        mode = 'Evaluate'
    else:
        model.train()
        mode = 'Train'

    start = time.time()
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, (features, labels) in t:
            # measure data loading time
            data_time.update(time.time() - start)
            
            labels = labels.cuda()
            if isinstance(features, list):
                features = [f.cuda() for f in features]
                output = model(features[0]) if len(features) == 1 else model(features)
            else:
                features = features.cuda()
                output = model(features)
            
            loss = criterion(labels, output)
            losses.update(loss.item(), features[0].size(0))

            if not evaluate_:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            t.set_description(description.format(mode=mode,
                                                 epoch=epoch,
                                                 avg_time=batch_time.avg,
                                                 it_time=batch_time.val,
                                                 avg_data_time=data_time.avg,
                                                 it_data_time=data_time.val,
                                                 loss=losses.avg))

    return losses.avg


def test(dataloader, model):
    batch_time, data_time = AverageMeter(), AverageMeter()
    description = 'Test (cls): ' \
                  'Time {avg_time:.3f}s (it:{it_time:.3f}s) ' \
                  'Data:{avg_data_time:.3f}s (it:{it_data_time:.3f}s) '
    model.eval()
    start_time = time.time()
    all_labels, all_outputs = [], []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (features, labels) in t:
            # measure data loading time
            data_time.update(time.time() - start_time)

            if isinstance(features, list):
                features = [f.cuda() for f in features]
                output = model(features[0]) if len(features) == 1 else model(features)
            else:
                features = features.cuda()
                output = model(features)

            all_labels.append(labels.detach().numpy())
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            t.set_description(description.format(avg_time=batch_time.avg,
                                                 it_time=batch_time.val,
                                                 avg_data_time=data_time.avg,
                                                 it_data_time=data_time.val))

    AP = []
    for i in range(1, dataloader.dataset.num_classes):
        AP.append(avg_prec_score(np.concatenate(all_labels)[:, i],
                                 np.concatenate(all_outputs)[:, i]))
    return np.mean(AP)
