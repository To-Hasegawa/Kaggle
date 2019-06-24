from common import *

from model import *
from dataset import *



def do_valid(net, valid_loader):

    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node,edge,edge_index,node_batch_index,
            coupling_index,coupling_type,coupling_value,coupling_batch_index,
            infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index  = edge_index.cuda()
        node_batch_index = node_batch_index.cuda()
        coupling_index = coupling_index.cuda()
        coupling_type  = coupling_type.cuda()
        coupling_value = coupling_value.cuda()
        coupling_batch_index = coupling_batch_index.cuda()

        with torch.no_grad():
            predict = net(node,edge,edge_index,node_batch_index, coupling_index,coupling_type,coupling_batch_index)
            loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_type.data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_num  += batch_size

        print('\r %8d/%8d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))
    #print('')
    valid_loss = valid_loss/valid_num

    #compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type  = np.concatenate(valid_coupling_type).astype(np.int32)

    mae, log_mae = compute_kaggle_metric( predict,
                    coupling_value, coupling_type, len(COUPLING_TYPE))


    valid_loss=[
        valid_loss, mae, log_mae
    ]
    return valid_loss


def run_train():

    out_dir = \
        '/root/share/project/kaggle/2019/champs_scalar/result/kaggle_predict5.1-a'

    initial_checkpoint = \
        '/root/share/project/kaggle/2019/champs_scalar/result/kaggle_predict5.1/checkpoint/00030000_model.pth' #None
        #None #out_dir + '/checkpoint/00020000_model.pth' #None


    schduler = NullScheduler(lr=0.0001)

    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 32 #*2 #280*2 #256*4 #128 #256 #512  #16 #32


    train_dataset = ChampsDataset(
                csv='train',
                mode ='train',
                #split='debug_split_by_mol.1000.npy', #
                split='train_split_by_mol.80003.npy',
                augment=None,
    )
    train_loader  = DataLoader(
                train_dataset,
                #sampler     = SequentialSampler(train_dataset),
                sampler     = RandomSampler(train_dataset),
                batch_size  = batch_size,
                drop_last   = True,
                num_workers = 16,
                pin_memory  = True,
                collate_fn  = null_collate
    )

    valid_dataset = ChampsDataset(
                csv='train',
                mode='train',
                #split='debug_split_by_mol.1000.npy', # #,None
                split='valid_split_by_mol.5000.npy',
                augment=None,
    )
    valid_loader = DataLoader(
                valid_dataset,
                #sampler     = SequentialSampler(valid_dataset),
                sampler     = RandomSampler(valid_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 8,
                pin_memory  = True,
                collate_fn  = null_collate
    )


    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('\n')


    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    iter_accum  = 1
    num_iters   = 3000  *1000
    iter_smooth = 50
    iter_log    = 500
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 2500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']

            optimizer.load_state_dict(checkpoint['optimizer'])
        pass



    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('                        |----------- VALID --------|-- TRAIN/BATCH --|         \n')
    log.write('rate       iter  epoch  | loss      mae    log_mae |  loss           |  time   \n')
    log.write('------------------------------------------------------------------------------------\n')
              #0.00100     1.0    32.0 | 4.84901   1.466   +0.314 |  5.62088        |  0 hr 04 min

    train_loss   = np.zeros(16,np.float32)
    valid_loss   = np.zeros(16,np.float32)
    batch_loss   = np.zeros(16,np.float32)
    iter = 0
    i    = 0


    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(16,np.float32)
        sum = 0

        optimizer.zero_grad()
        for node,edge,edge_index,node_batch_index, coupling_index,coupling_type,coupling_value,coupling_batch_index, infor in train_loader:

            #while 1:
                batch_size = len(infor)
                iter  = i + start_iter
                epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


                # debug-----------------------------
                if 0:
                    pass

                #if 0:
                if (iter % iter_valid==0):
                    valid_loss = do_valid(net, valid_loader) # pass #



                if (iter % iter_log==0):
                    print('\r',end='',flush=True)
                    asterisk = '*' if iter in iter_save else ' '
                    log.write('%0.5f %7.1f%s %6.1f | %0.5f   %0.3f   %+0.3f | %7.5f        | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             valid_loss[0], valid_loss[1], valid_loss[2],
                             train_loss[0],
                             time_to_str((timer() - start),'min'))
                    )
                    log.write('\n')

                #if 0:
                if iter in iter_save:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                    pass




                # learning rate schduler -------------
                lr = schduler(iter)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr)
                rate = get_learning_rate(optimizer)

                # one iteration update  -------------
                #net.set_mode('train',is_freeze_bn=True)

                net.train()
                node = node.cuda()
                edge = edge.cuda()
                edge_index  = edge_index.cuda()
                node_batch_index = node_batch_index.cuda()
                coupling_index = coupling_index.cuda()
                coupling_type  = coupling_type.cuda()
                coupling_value = coupling_value.cuda()
                coupling_batch_index = coupling_batch_index.cuda()

                predict = net(node,edge,edge_index,node_batch_index, coupling_index,coupling_type,coupling_batch_index)
                loss = criterion(predict, coupling_value)

                (loss/iter_accum).backward()
                if (iter % iter_accum)==0:
                    optimizer.step()
                    optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:3] = np.array(( loss.item(),0,0,))
                sum_train_loss += batch_loss
                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum_train_loss = np.zeros(16,np.float32)
                    sum = 0


                print('\r',end='',flush=True)
                asterisk = ' '
                print('%0.5f %7.1f%s %6.1f | %7.5f   %0.3f   %+0.3f | %0.5f        | %s' % (\
                             rate, iter/1000, asterisk, epoch,
                             valid_loss[0], valid_loss[1], valid_loss[2],
                             batch_loss[0],
                             time_to_str((timer() - start),'min'))
                , end='',flush=True)
                i=i+1


        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')


'''

0.00000     0.0*    0.0 | 1689.17774  21.678 2.079 | 0.00000  |  0 hr 00 min
0.00100     0.5    64.0 | 1689.17774  21.678 2.079 | 10.62004  |  0 hr 02 min
0.00100     1.0   128.0 | 6.57140  1.711 0.469 | 6.80375  |  0 hr 05 min
0.00100     1.5   192.0 | 6.57140  1.711 0.469 | 6.21540  |  0 hr 08 min
0.00100     2.0   256.0 | 4.32079  1.367 0.246 | 4.23666  |  0 hr 11 min


champs v5_1
0.00000     0.0*    0.0 | 1693.19490  21.706 2.084 | 0.00000  |  0 hr 00 min
0.00100     0.5    16.0 | 7.41601  1.942 0.617 | 10.70206  |  0 hr 02 min
0.00100     1.0    32.0 | 4.84901  1.466 0.314 | 5.62088  |  0 hr 04 min
0.00100     1.5    48.0 | 3.52605  1.225 0.144 | 5.11197  |  0 hr 06 min
0.00100     2.0    64.0 | 3.30143  1.140 0.044 | 3.79732  |  0 hr 08 min
0.00100     2.5    80.0 | 2.23965  0.964 -0.074 | 4.63638  |  0 hr 10 min
0.00100     3.0    96.0 | 1.92562  0.939 -0.129 | 2.53014  |  0 hr 12 min
0.00100     3.5   112.0 | 2.13149  0.913 -0.123 | 2.38103  |  0 hr 14 min
0.00100     4.0   128.0 | 1.38566  0.714 -0.360 | 2.30549  |  0 hr 16 min
'''
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')