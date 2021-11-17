import sys
import os
import argparse
import logging
import json
import time
import numpy as np
import torch
import sklearn.metrics as skmetrics
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
from torchvision import models
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from data.image_producer_train import ImageDatasetTrain
from data.image_producer_val import ImageDatasetVal
from data.image_producer_test import ImageDatasetTest
from data.se_resnet import se_resnet50

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--config_name', default=None, metavar='CONFIG_NAME', type=str,
                    help='Name of the config name that required')
parser.add_argument('--save_path_name', default=None, metavar='SAVE_PATH_NAME', type=str,
                    help='Path to the saved models')
parser.add_argument('--mode', default='train', metavar='SAVE_PATH_NAME', type=str,
                    help='Phase that doing training or testing')
parser.add_argument('--test_module_metric', default='acc', metavar='TEST_MODULE_METRIC', type=str,
                    help='Metric based module choosing for test phase, loss/acc/sen')
parser.add_argument('--num_workers', default=24, type=int, help='number of workers for each data loader, default 24.')
args = parser.parse_args()

# CPU/GPU切换
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 创建文件保存文件夹
curr_path = os.getcwd()
configs_path = os.path.join(curr_path, 'configs')
results_path = os.path.join(curr_path, 'results')
if not os.path.exists(results_path):
    os.mkdir(results_path)
save_path = os.path.join(results_path, args.save_path_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)


def chose_model(cnn):
    if cnn['model'] == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif cnn['model'] == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif cnn['model'] == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif cnn['model'] == 'se50':
        model = se_resnet50(pretrained=False)
    else:
        raise Exception("I have not add any models. ")
    return model


# ACC, B_Acc, cohen-kappa, Sensitivity/recall, Specificity, AUC
def metrics_predict(y_true, y_pred):
    acc = skmetrics.accuracy_score(y_true, y_pred)
    b_acc = skmetrics.balanced_accuracy_score(y_true, y_pred)
    k = skmetrics.cohen_kappa_score(y_true, y_pred)
    sensitivity = skmetrics.recall_score(y_true, y_pred)
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return acc, b_acc, k, sensitivity, specificity


def metrics_score(y_true, y_pred, is_test=False):
    fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
    auc = skmetrics.auc(fpr, tpr)
    if is_test:
        return auc, fpr, tpr
    else:
        return auc


def train_epoch(summary, summary_writer, cnn, model, loss_fn, optimizer, dataloader_train):
    # 将模型设置为训练模式
    model.train()
    loss_sum = 0
    probs_list = []
    predicts_list = []
    target_train_list = []
    # dataloader本质是一个可迭代对象, 根据设置的batch大小对原始数据进行切分
    # 使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问
    # 也可以使用`for inputs, labels in dataloaders`进行可迭代对象的访问
    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    dataiter_train = iter(dataloader_train)

    time_now = time.time()
    for step in range(steps):
        data_train, target_train = next(dataiter_train)
        data_train = Variable(data_train.float().to(device, non_blocking=True))
        target_train = Variable(target_train.float().to(device, non_blocking=True))

        output = model(data_train)
        output = torch.squeeze(output)  # noqa
        loss = loss_fn(output, target_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = output.sigmoid()
        predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
        prob = probs.cpu().data.numpy().tolist()
        predict = predicts.cpu().data.numpy().tolist()
        target = target_train.cpu().data.numpy().tolist()

        probs_list.extend(prob)
        predicts_list.extend(predict)
        target_train_list.extend(target)

        loss_data = loss.data
        summary['step'] += 1
        loss_sum += loss_data

        # 在每个epoch中，每隔n个step进行一次记录
        # if summary['step'] % cnn['log_every'] == 0:
        #     summary_writer.add_scalar('train/loss', loss_data, summary['step'])
        #     summary_writer.add_scalar('train/acc', acc, summary['step'])
        #     summary_writer.add_scalar('train/b_acc', b_acc, summary['step'])
        #     summary_writer.add_scalar('train/k_score', k_score, summary['step'])
        #     summary_writer.add_scalar('train/sen', sensitivity, summary['step'])
        #     summary_writer.add_scalar('train/spe', specificity, summary['step'])
        #     summary_writer.add_scalar('train/auc', auc, summary['step'])

    time_spent = time.time() - time_now
    probs_array = np.array(probs_list)
    predicts_array = np.array(predicts_list)
    target_train_array = np.array(target_train_list)
    acc, b_acc, k_score, sensitivity, specificity = metrics_predict(target_train_array, predicts_array)
    auc = metrics_score(target_train_array, probs_array)
    logging.info(
        '{}, Epoch : {}, Training Loss : {:.5f}, Training Acc : {:.4f}, Training Balanced Acc : {:.4f}, Training K Score : {:.4f}, Training Sensitivity : {:.4f}, Training Specificity : {:.4f}, Training AUC : {:.4f}, Run Time : {:.2f}'
        .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1, loss_data, acc, b_acc, k_score, sensitivity, specificity, auc, time_spent))

    summary['epoch'] += 1
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc
    summary['b_acc'] = b_acc
    summary['k_score'] = k_score
    summary['sen'] = sensitivity
    summary['spe'] = specificity
    summary['auc'] = auc

    return summary


def valid_epoch(summary, model, loss_fn, dataloader_valid):
    with torch.no_grad():
        model.eval()

        steps = len(dataloader_valid)
        batch_size = dataloader_valid.batch_size
        dataiter_valid = iter(dataloader_valid)

        loss_sum = 0
        probs_list = []
        predicts_list = []
        target_valid_list = []

        for step in range(steps):
            data_valid, target_valid = next(dataiter_valid)

            data_valid = Variable(data_valid.float().to(device, non_blocking=True))
            target_valid = Variable(target_valid.float().to(device, non_blocking=True))

            output = model(data_valid)
            output = torch.squeeze(output)  # important
            loss = loss_fn(output, target_valid)

            probs = output.sigmoid()
            predicts = (probs >= 0.5).type(torch.cuda.FloatTensor)
            probs = probs.cpu().data.numpy().tolist()
            predicts = predicts.cpu().data.numpy().tolist()
            target_valid = target_valid.cpu().data.numpy().tolist()

            probs_list.extend(probs)
            predicts_list.extend(predicts)
            target_valid_list.extend(target_valid)

            loss_data = loss.data
            loss_sum += loss_data

        probs_array = np.array(probs_list)
        predicts_array = np.array(predicts_list)
        target_valid_array = np.array(target_valid_list)

        acc, b_acc, k_score, sensitivity, specificity = metrics_predict(target_valid_array, predicts_array)
        if args.mode == 'test':
            auc, fpr, tpr = metrics_score(target_valid_array, probs_array, True)
            np.save(os.path.join(save_path, "fpr.npy"), fpr)
            np.save(os.path.join(save_path, "tpr.npy"), tpr)
        else:
            auc = metrics_score(target_valid, probs)
        summary['loss'] = loss_sum / steps
        summary['acc'] = acc
        summary['b_acc'] = b_acc
        summary['k_score'] = k_score
        summary['sen'] = sensitivity
        summary['spe'] = specificity
        summary['auc'] = auc

        return summary


def run(args):
    # 网络及数据等相关配置读取
    if args.mode == 'train':
        config_file_path = os.path.join(configs_path, args.config_name)
    elif args.mode == 'test':
        config_file_path = os.path.join(save_path, 'cnn.json')
    else:
        raise Exception("arg mode input error")
    with open(config_file_path, 'r') as f:
        cnn = json.load(f)

    # 将当前使用的配置文件复制到存储文件夹下进行备份
    if args.mode == 'train':
        with open(os.path.join(save_path, 'cnn.json'), 'w') as f:
            json.dump(cnn, f, indent=1)

    # GPU使用设置
    num_GPU = torch.cuda.device_count()
    if num_GPU > 0:
        batch_size_train = cnn['batch_size'] * num_GPU
        batch_size_valid = cnn['batch_size'] * num_GPU
        num_workers = args.num_workers * num_GPU
    else:
        batch_size_train = cnn['batch_size']
        batch_size_valid = cnn['batch_size']
        num_workers = args.num_workers

    # 模型框架选择
    model = chose_model(cnn)

    # 模型网络修改（根据实际情况）
    if cnn['model'] == 'vgg16':
        model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
    else:
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 1)

    # 模型并行设置及GPU使用，当device_ids设置为None时，函数会调用torch.cuda.device_count()搜索可使用gpu的数量
    if num_GPU > 1:
        model = DataParallel(model, device_ids=None)
    model = model.to(device)

    # 损失函数选取以及优化器选择
    loss_fn = BCEWithLogitsLoss().to(device)

    # 训练模式
    if args.mode == 'train':
        epoches = cnn['epoch']
        optimizer = SGD(model.parameters(), lr=cnn['lr'], momentum=cnn['momentum'], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max = epoches)

        # 训练集数据加载
        dataset_train = ImageDatasetTrain(cnn['data_path_train'],
                                     cnn['image_size'],
                                     cnn['crop_size'],
                                     cnn['normalize'])

        # 验证集数据加载
        dataset_valid = ImageDatasetVal(cnn['data_path_valid'],
                                     cnn['image_size'],
                                     cnn['crop_size'],
                                     cnn['normalize'])

        # 训练数据dataload
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size_train,
                                      num_workers=num_workers)

        # 验证数据dataload
        dataloader_valid = DataLoader(dataset_valid,
                                      batch_size=batch_size_valid,
                                      num_workers=num_workers)

        summary_train = {'epoch': 0, 'step': 0}
        summary_valid = {'loss': float('inf'), 'acc': 0}
        summary_writer = SummaryWriter(save_path)
        loss_valid_best = float('inf')
        acc_valid_best = 0
        sen_valid_best = 0
        auc_valid_best = 0

        # 网络训练时间记录
        train_start_time = time.time()
        for epoch in range(epoches):
            summary_train = train_epoch(summary_train, summary_writer, cnn, model, loss_fn, optimizer, dataloader_train)

            # 根据是否并行进行相应的模型存储
            if num_GPU > 1:
                torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                            'state_dict': model.module.state_dict()}, os.path.join(save_path, 'train.ckpt'))
            else:
                torch.save(
                    {'epoch': summary_train['epoch'], 'step': summary_train['step'], 'state_dict': model.state_dict()},
                    os.path.join(save_path, 'train.ckpt'))

            time_now = time.time()
            summary_valid = valid_epoch(summary_valid, model, loss_fn, dataloader_valid)
            time_spent = time.time() - time_now

            # 学习率衰减
            scheduler.step()

            logging.info(
                '{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, Validation ACC: {:.4f}, Validation Balanced Acc : {:.4f}, Validation K Score : {:.4f}, Validation Sensitivity : {:.4f}, Validation Specificity : {:.4f}, Validation AUC : {:.4f}, Run Time: {:.2f}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'], summary_train['step'],
                        summary_valid['loss'], summary_valid['acc'], summary_valid['b_acc'], summary_valid['k_score'],
                        summary_valid['sen'], summary_valid['spe'], summary_valid['auc'], time_spent))

            # tensorboard记录
            # 训练数据
            summary_writer.add_scalar('train/loss', summary_train['loss'], summary_train['epoch'])
            summary_writer.add_scalar('train/acc', summary_train['acc'], summary_train['epoch'])
            summary_writer.add_scalar('train/b_acc', summary_train['b_acc'], summary_train['epoch'])
            summary_writer.add_scalar('train/k_score', summary_train['k_score'], summary_train['epoch'])
            summary_writer.add_scalar('train/sen', summary_train['sen'], summary_train['epoch'])
            summary_writer.add_scalar('train/spe', summary_train['spe'], summary_train['epoch'])
            summary_writer.add_scalar('train/auc', summary_train['auc'], summary_train['epoch'])
            # 验证数据
            summary_writer.add_scalar('valid/loss', summary_valid['loss'], summary_train['epoch'])
            summary_writer.add_scalar('valid/acc', summary_valid['acc'], summary_train['epoch'])
            summary_writer.add_scalar('valid/b_acc', summary_valid['b_acc'], summary_train['epoch'])
            summary_writer.add_scalar('valid/k_score', summary_valid['k_score'], summary_train['epoch'])
            summary_writer.add_scalar('valid/sen', summary_valid['sen'], summary_train['epoch'])
            summary_writer.add_scalar('valid/spe', summary_valid['spe'], summary_train['epoch'])
            summary_writer.add_scalar('valid/auc', summary_valid['auc'], summary_train['epoch'])

            # 以验证集上loss, acc, sen最好结果保存训练模型
            if num_GPU > 1:
                if summary_valid['loss'] <= loss_valid_best:
                    loss_valid_best = summary_valid['loss']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.module.state_dict()}, os.path.join(save_path, 'best_loss.ckpt'))

                if summary_valid['acc'] >= acc_valid_best:
                    acc_valid_best = summary_valid['acc']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.module.state_dict()}, os.path.join(save_path, 'best_acc.ckpt'))

                if summary_valid['sen'] >= sen_valid_best:
                    sen_valid_best = summary_valid['sen']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.module.state_dict()}, os.path.join(save_path, 'best_sen.ckpt'))
                
                if summary_valid['auc'] >= auc_valid_best:
                    auc_valid_best = summary_valid['auc']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.module.state_dict()}, os.path.join(save_path, 'best_auc.ckpt'))
            else:
                if summary_valid['loss'] <= loss_valid_best:
                    loss_valid_best = summary_valid['loss']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.state_dict()}, os.path.join(save_path, 'best_loss.ckpt'))

                if summary_valid['acc'] >= acc_valid_best:
                    acc_valid_best = summary_valid['acc']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.state_dict()}, os.path.join(save_path, 'best_acc.ckpt'))

                if summary_valid['sen'] >= sen_valid_best:
                    sen_valid_best = summary_valid['sen']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.state_dict()}, os.path.join(save_path, 'best_sen.ckpt'))
                
                if summary_valid['auc'] >= auc_valid_best:
                    auc_valid_best = summary_valid['auc']

                    torch.save({'epoch': summary_train['epoch'], 'step': summary_train['step'],
                                'state_dict': model.state_dict()}, os.path.join(save_path, 'best_auc.ckpt'))

        train_spent_time = time.time() - train_start_time
        with open(os.path.join(save_path, 'final_results.txt'), "a+") as f:
            f.write('Training time cost: ' + str(train_spent_time) + '\n')
        summary_writer.close()
    else:
        # 参数加载
        checkpoint = torch.load(os.path.join(save_path, 'best_' + args.test_module_metric + '.ckpt'))#'best_' + args.test_module_metric + '.ckpt'
        if num_GPU > 1:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

        # 测试集数据加载
        dataset_test = ImageDatasetTest(cnn['data_path_test'], cnn['image_size'], cnn['crop_size'], cnn['normalize'])

        # 测试数据dataload
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size_valid, num_workers=num_workers)

        # 结果获取
        summary_test = {}
        summary_test = valid_epoch(summary_test, model, loss_fn, dataloader_test)

        # 最终结果文本记录
        with open(os.path.join(save_path, 'final_results.txt'), "a+") as f:
            f.write('-----------------------------------------------' + '\n')
            f.write(args.test_module_metric + ': ' + '\n')
            for key in summary_test:
                f.write(key + ': ' + str(summary_test[key]) + '\n')
            f.write('-----------------------------------------------' + '\n')


def main():
    logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
