# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from models.resnet12_2 import resnet12
from models.MetaNODE_Model import DiffNODEOPTClassifier

from utils import set_gpu, Timer, count_accuracy, check_dir, log

def one_hot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ResNet':
        network = resnet12().cuda()
        network = torch.nn.DataParallel(network)
    else:
        print ("Cannot recognize the network type")
        assert(False)
    cls_head = DiffNODEOPTClassifier().cuda()
    return (network, cls_head)

def get_dataset(options):
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

def test(opt, n_iter, dataset_test, data_loader):
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,
        nTestNovel=opt.val_query * opt.test_way,
        nTestBase=0,
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,
    )

    set_gpu(opt.gpu)
    check_dir(opt.load_path)

    log_file_path = os.path.join(opt.load_path, "test_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.load_path, 'model.pth'))
    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    state_dict = cls_head.state_dict()
    new_state_dict = {k: saved_models['head'][k] for k,v in state_dict.items()}
    cls_head.load_state_dict(new_state_dict)
    cls_head.eval()

    x_entropy = torch.nn.CrossEntropyLoss()

    _, _ = [x.eval() for x in (cls_head, embedding_net)]

    test_accuracies = []
    test_losses = []
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query

        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

        logit_querys = cls_head(emb_query, emb_support, labels_support, labels_query, opt.test_way, opt.val_shot, is_train=False, n_iter=n_iter)
        logit_query = logit_querys[-1]
        loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
        acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

        test_accuracies.append(acc.item())
        test_losses.append(loss.item())

    test_acc_avg = np.mean(np.array(test_accuracies))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    test_loss_avg = np.mean(np.array(test_losses))

    log(log_file_path, 'Test Loss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
        .format(test_loss_avg, test_acc_avg, test_acc_ci95))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=1,
                            help='number of support examples per validation class')
    parser.add_argument('--val-episode', type=int, default=600,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--load-path', default='./checkpoints')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ResNet')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet')


    opt = parser.parse_args()
    
    (dataset_test, data_loader) = get_dataset(opt)

    test(opt, 0, dataset_test, data_loader)