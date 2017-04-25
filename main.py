import huva.th_util as thu
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from pprint import pprint, pformat
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name',  type=str, default='')
parser.add_argument('-m', '--mode',  type=str, default='')
parser.add_argument('-o', '--optimizer',    type=str, default='SGD')
parser.add_argument('-b', '--batch-size',   type=int, default=128)
parser.add_argument('-nb','--num-base',     type=int, default=32)
parser.add_argument('-L1','--L1-decay',     type=float, default=0.0005*0.01)
parser.add_argument('-wd','--weight-decay', type=float, default=0.0005)
parser.add_argument('--momentum',           type=float, default=0.0)
parser.add_argument('--last2-mult',         type=float, default=1.0)
parser.add_argument('--last1-mult',         type=float, default=1.0)
parser.add_argument('--topo-base',          type=float, default=2.0)
parser.add_argument('-lr','--learning-rate',type=float, default=0.1) # default for SGD
parser.add_argument('--logfile',     type=str, default='')
parser.add_argument('--graphfolder', type=str, default='')
parser.add_argument('--force-name', action='store_true')
parser.add_argument('--auto-start', action='store_true')
args = parser.parse_args()

def make_data(batch_size=128):
    global dataset, loader, dataset_test, loader_test
    (dataset, loader), (dataset_test, loader_test) = thu.make_data_cifar10(batch_size)

def make_model(base=32, num_class=10):
    global model, model_conf
    bn = base # shorter name
    model_conf = [
        ('input',   (3,   None)),
        ('conv1_1', (bn*1,None)),
        ('conv1_2', (bn*1,None)),
        ('pool1'  , (2,   2)),
        ('conv2_1', (bn*2,None)),
        ('conv2_2', (bn*2,None)),
        ('pool2'  , (2,   2)),
        ('conv3_1', (bn*4,None)),
        ('conv3_2', (bn*4,None)),
       #('conv3_3', (bn*4,None)),
        ('pool3'  , (2,   2)),
        ('conv4_1', (bn*8,None)),
        ('conv4_2', (bn*8,None)),
       #('conv4_3', (bn*8,None)),
        ('pool4'  , (2,   2)),
        ('conv5_1', (bn*8,None)),
        ('conv5_2', (bn*8,None)),
       #('conv5_3', (bn*8,None)),
        ('pool5'  , (2,   2)),
        ('fc6'    , (bn*8,None)),
        ('logit'  , (num_class, None)),
        ('flatter', (None,None)),
    ]
    model = thu.make_cnn_with_conf(model_conf)
    model = model.cuda()

def make_param_groups_with_proportional_variance(model, mode='var', base_decay=0.0005 * 0.0001, dup_mult=4.0): 
    # 0.0005 is conventional decay, 0.01 is conventional std
    """ estimate W variance for every conv layer """
    param_groups = []
    duplicate_multiplier = 1.0
    for layer in model.children():
        conv_layer = None
        if isinstance(layer, nn.Conv2d):
            conv_layer = layer
        elif isinstance(layer, nn.Sequential):
            conv_layer = layer[0]
        elif isinstance(layer, nn.MaxPool2d):
            assert layer.kernel_size==2 and layer.stride==2
            duplicate_multiplier /= dup_mult # number of duplication due to convolution reduced by 4
        else:
            print("Ignored child type: {}".format(type(layer)))
        if conv_layer is None:
            continue
        assert isinstance(conv_layer, nn.Conv2d)
        if mode=='var':
            """
            decay proportional to inverse_variance * num_duplication
                  proportional to in_channels*k*k  * num_duplication
            Across every MP, in_channel increase by 2, num_duplication reduce by 4, so total reduce by 2
            But.. should we account for effect of MP on backprop?
            """
            wd = base_decay / conv_layer.weight.data.var()
        elif mode=='std':
            wd = base_decay / conv_layer.weight.data.std()
        else:
            assert False, 'unknown normalization mode: {}'.format(mode)
        wd *= duplicate_multiplier
        param_groups.append({'params':layer.parameters(), 'weight_decay':wd})
    return param_groups

def show_optimizer():
    for group in optimizer.param_groups:
        logger.log(group['weight_decay'])

def make_optimizer():
    global optimizer, criterion
    lr = args.learning_rate

    if args.optimizer=='Adam':
        lr = 0.001 # ignore lr advice
        optimizer = thu.MonitoredAdam(model.parameters(), lr, weight_decay=args.weight_decay)

    elif args.optimizer=='SGD':
        optimizer = thu.MonitoredSGD(model.parameters(), lr, weight_decay=args.weight_decay, momentum=args.momentum)

    elif args.optimizer=='SGDL1':
        params   = {id(p):p for p in model.parameters()}
        params_conv = sum(map(lambda l:list(l.parameters()), 
                            [model.conv1_1, model.conv1_2, 
                             model.conv2_1, model.conv2_2, 
                             model.conv3_1, model.conv3_2,
                             model.conv4_1, model.conv4_2]),[])
        for p in params_conv: del params[id(p)]
        params_rest = params.values()
        param_groups = [
                {'params':params_conv, 'weight_decay': args.L1_decay, 'L1': True}, # L1 decay
                {'params':params_rest, 'weight_decay': args.weight_decay}          # L2 decay
            ]
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)

    elif args.optimizer=='SGDL1A':
        param_groups = [ {'params':model.parameters(), 'weight_decay': args.L1_decay, 'L1': True} ]
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)

    elif args.optimizer in ['SGD_var_dup','SGD_var_dup4']:
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=4.0)
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)

    elif args.optimizer=='SGD_var_dup2':
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)

    elif args.optimizer=='SGD_var_dup2_last2_mult':
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last2_mult
        param_groups[-2]['weight_decay'] *= args.last2_mult
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)

    elif args.optimizer=='SGD_var_dup2_last1_mult':
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last1_mult
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)

    elif args.optimizer in ['SGD_topo.a', 'SGD_var_dup2_last1_mult_topo.a']:
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last1_mult
        fc6_decay_mult = thu.get_topographic_decay_multiplier(model.fc6[0], mults=[1,args.topo_base,args.topo_base**2])
        param_groups[-2]['decay_mults'] = [fc6_decay_mult, None, None] # conv, bn, relu
        optimizer = thu.MonitoredSpecificSGD(param_groups, lr, momentum=args.momentum, specific_mode=None)

    elif args.optimizer in ['SGD_topo.b', 'SGD_var_dup2_last1_mult_topo.b']:
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last1_mult
        conv5_2_decay_mult = thu.get_topographic_decay_multiplier(model.conv5_2[0], mults=[1,args.topo_base,args.topo_base**2])
        param_groups[-3]['decay_mults'] = [conv5_2_decay_mult, None, None] # conv, bn, relu
        optimizer = thu.MonitoredSpecificSGD(param_groups, lr, momentum=args.momentum, specific_mode=None)

    elif args.optimizer in ['SGD_topo.ab', 'SGD_var_dup2_last1_mult_topo.ab']:
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last1_mult
        fc6_decay_mult     = thu.get_topographic_decay_multiplier(model.fc6[0], mults=[1,args.topo_base,args.topo_base**2])
        conv5_2_decay_mult = thu.get_topographic_decay_multiplier(model.conv5_2[0], mults=[1,args.topo_base,args.topo_base**2])
        param_groups[-2]['decay_mults'] = [fc6_decay_mult,     None, None] # conv, bn, relu
        param_groups[-3]['decay_mults'] = [conv5_2_decay_mult, None, None] # conv, bn, relu
        optimizer = thu.MonitoredSpecificSGD(param_groups, lr, momentum=args.momentum, specific_mode=None)

    elif args.optimizer=='LSSGD_var_dup2_last1_mult': # layer-specific SGD
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last1_mult
        optimizer = thu.MonitoredSpecificSGD(param_groups, lr, momentum=args.momentum, specific_mode='layer')

    elif args.optimizer=='LSSGD_var_dup2_last1_mult_nwm': # layer-specific SGD, no-weight-norm-multiplier
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=2.0)
        param_groups[-1]['weight_decay'] *= args.last1_mult
        optimizer = thu.MonitoredSpecificSGD(param_groups, lr, momentum=args.momentum, specific_mode='layer', mult_weight_norm=False)

    elif args.optimizer=='SGD_var_dup1':
        param_groups = make_param_groups_with_proportional_variance( model, base_decay=args.weight_decay, dup_mult=1.0)
        optimizer = thu.MonitoredSGD(param_groups, lr, momentum=args.momentum)
    else:
        assert False, 'unknown optimizer mode: {}'.format(args.optimizer)
    criterion = nn.CrossEntropyLoss().cuda()

def make_all():
    global logger
    if args.graphfolder == '': 
        args.graphfolder = 'logs/{}_graphs/'.format(args.name)
    if args.logfile == '':
        args.logfile = 'logs/{}.log'.format(args.name)
    if not args.force_name and os.path.exists(args.logfile):
        assert False, 'abort because {} already exists'.format(args.logfile)
    logger = thu.LogPrinter(args.logfile)
    make_data(args.batch_size)
    make_model(args.num_base)
    model.args = args
    make_optimizer()
    logger.log(str(model))
    logger.log(pformat(args.__dict__))

min_loss = 999999
min_loss_batches = 0
epoch_trained = 0
max_accuracy = 0
max_accuracy_epoch = None

def train(num_epochs, report_interval=30):
    global epoch_trained, min_loss, min_loss_batches, max_accuracy, max_accuracy_epoch
    for epoch in xrange(num_epochs):
        model.train()
        gloss = 0
        num_correct = 0
        num_samples = 0
        for batch, (imgs, labs) in enumerate(loader):
            should_report = (batch+1) % report_interval == 0
            should_monitor = should_report or (batch+1) == len(loader)
            """ forward """
            v_imgs = Variable(imgs.cuda())
            v_labs = Variable(labs.cuda())
            v_outs = model(v_imgs)
            v_loss = criterion(v_outs, v_labs)
            """ backward """
            optimizer.zero_grad()
            v_loss.backward() # we want to perform maximization, so negate the ones
            optimizer.step(update_monitor=should_monitor)
            """ report """
            num_correct += thu.get_num_correct(v_outs.data.cpu(), labs)
            num_samples += imgs.size(0)
            gloss += v_loss.data[0]
            min_loss_batches += 1
            if should_report:
                avg_loss = gloss / report_interval
                gloss = 0
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    min_loss_batches = 0
                logger.log("{:3d} {:3d} {:3d}, {:5f} [{:5f}/{:5f}] [{:4d}/{:5f}]".format(
                    epoch_trained, epoch, batch+1, avg_loss, 
                    optimizer.update_norm, thu.get_model_param_norm(model),
                    min_loss_batches, min_loss))
        """ show a glimpse of the monitored norms recorded in previous batch """
        logger.log("Showing Monitored Norms")
        w_norms = []
        g_norms = []
        mg_norms= []
        g_proj_ds= []
        d_norms = []
        u_norms = []
        for group in optimizer.param_groups:
            for norm, p in zip(group['norms'], group['params']):
                if p.dim() == 1: continue
                w_norms. append(norm.w_norm)
                g_norms. append(norm.g_norm)
                mg_norms.append(norm.mg_norm)
                g_proj_ds.append(norm.g_proj_d)
                d_norms. append(norm.d_norm)
                u_norms. append(norm.u_norm)
        format_string = '{:7f}, '*len(w_norms)
        logger.log(format_string.format(* w_norms))
        logger.log(format_string.format(* g_norms))
        if optimizer.param_groups[0]['momentum'] != 0:
            logger.log(format_string.format(*mg_norms))
        logger.log(format_string.format(* g_proj_ds))
        logger.log(format_string.format(* d_norms))
        logger.log(format_string.format(* u_norms))
        """ show evaluation statistics """
        logger.log("Epoch {} done. Evaluation:".format(epoch))
        logger.log((num_correct, num_samples))
        num_test_correct, num_test_samples = eval_accuracy(loader_test)
        logger.log((num_test_correct, num_test_samples))
        if num_test_correct > max_accuracy:
            max_accuracy = num_test_correct
            max_accuracy_epoch = (epoch_trained, epoch)
            logger.log('Maximum test accuracy increased to: {}'.format(max_accuracy))
    epoch_trained += 1

def eval_accuracy(loader, testmodel=None, max_batches=99999999):
    if testmodel is None: testmodel = model
    testmodel.eval()
    num_correct = 0
    num_samples = 0
    for i, (imgs, labs) in enumerate(loader):
        if i >= max_batches:
            break
        v_imgs   = Variable(imgs).cuda()
        v_output = testmodel(v_imgs)
        output = v_output.data.cpu()
        num_correct += thu.get_num_correct(output, labs)
        num_samples += labs.size(0)
    return num_correct, num_samples

def decayed_training(schedule):
    global lr, logger
    lr = args.learning_rate
    logger.log('Decayed training with schedule: {}'.format(schedule))
    for epochs in schedule:
        train(epochs, 50)
        lr /= 2
        thu.set_learning_rate(optimizer, lr)
    save_model()

def save_layer_hist(layer):
    thu.collect_output_save_hist_for_layer(model, layer, loader_test, args.graphfolder)

def save_layer_gbp(layer):
    thu.collect_output_save_gbp_for_layer(model, layer, dataset_test, loader_test, args.graphfolder)

def save_graphs(hist=True, gbp=True):
    layers = [ model.conv1_1[0], model.conv2_1[0], model.conv3_1[0], model.conv4_1[0], 
               model.conv5_1[0], model.fc6[0],     model.logit ]
    for layer in layers:
        name = [name for name, module in model.named_modules() if module is layer][0]
        print('saving for {}'.format(name))
        if hist:
            save_layer_hist(layer)
        if gbp:
            save_layer_gbp(layer)

def save_model():
    torch.save(model, 'logs/{}.pth'.format(args.name))

def load_model(path):
    global model, args
    model = torch.load(path)
    args = model.args

if __name__=='__main__':
    if args.auto_start:
        make_all()
        show_optimizer()
        decayed_training([20]*10)
