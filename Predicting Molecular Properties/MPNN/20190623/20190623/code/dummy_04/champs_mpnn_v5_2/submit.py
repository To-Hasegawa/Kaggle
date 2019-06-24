from common import *

from model import *
from dataset import *
from data import *


def run_submit():

    out_dir = \
        '/root/share/project/kaggle/2019/champs_scalar/result/kaggle_predict5.1'

    initial_checkpoint = \
        '/root/share/project/kaggle/2019/champs_scalar/result/kaggle_predict5.1/checkpoint/00034000_model.pth' #None
        #None #out_dir + '/checkpoint/00020000_model.pth' #None


    csv_file = out_dir +'/submit/submit-%s.csv'%(initial_checkpoint.split('/')[-1][:-4])



    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 32 #*2 #280*2 #256*4 #128 #256 #512  #16 #32

    if 1:
        test_dataset = ChampsDataset(
                    mode ='test',
                    csv  ='test',
                    #split='debug_split_by_mol.1000.npy',
                    split=None,
                    augment=None,
        )
    if 0:
        test_dataset = ChampsDataset(
                    mode ='train',
                    csv  ='train',
                    #split='debug_split_by_mol.1000.npy',
                    split='valid_split_by_mol.5000.npy',
                    augment=None,
        )

    test_loader  = DataLoader(
                test_dataset,
                sampler     = SequentialSampler(test_dataset),
                #sampler     = RandomSampler(train_dataset),
                batch_size  = batch_size,
                drop_last   = False,
                num_workers = 0,
                pin_memory  = True,
                collate_fn  = null_collate
    )

    log.write('batch_size = %d\n'%(batch_size))
    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('\n')


    ## start testing here! ##############################################
    test_num = 0
    test_predict = []
    test_coupling_type  = []
    test_coupling_value = []
    test_id = []

    test_loss = 0

    start = timer()
    for b, (node,edge,edge_index,node_batch_index,
            coupling_index,coupling_type,coupling_value,coupling_batch_index,
            infor) in enumerate(test_loader):

        net.eval()
        with torch.no_grad():
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

        #---
        batch_size = len(infor)
        test_id.extend(list(np.concatenate([infor[b][2] for b in range(batch_size)])))

        test_predict.append(predict.data.cpu().numpy())
        test_coupling_type.append(coupling_type.data.cpu().numpy())
        test_coupling_value.append(coupling_value.data.cpu().numpy())

        test_loss += loss.item()*batch_size
        test_num += batch_size


        print('\r %8d/%8d     %0.2f  %s'%(
            test_num, len(test_dataset),test_num/len(test_dataset),
              time_to_str(timer()-start,'min')),end='',flush=True)


        pass  #-- end of one data loader --
    assert(test_num == len(test_dataset))
    print('\n')

    id       = test_id
    predict  = np.concatenate(test_predict)
    df = pd.DataFrame(list(zip(id, predict)), columns =['id', 'scalar_coupling_constant'])
    df.to_csv(csv_file,index=False)

    log.write('id        = %d\n'%len(id))
    log.write('predict   = %d\n'%len(predict))
    log.write('csv_file  = %s\n'%csv_file)

    #-------------------------------------------------------------
    # for debug
    if test_dataset.mode == 'train':
        test_loss = test_loss/test_num

        coupling_value = np.concatenate(test_coupling_value)
        coupling_type  = np.concatenate(test_coupling_type).astype(np.int32)

        mae, log_mae = compute_kaggle_metric( predict,
            coupling_value, coupling_type, len(COUPLING_TYPE))

        log.write('\n')
        log.write('test_loss =  %f\n'%test_loss)
        log.write('mae       =  %f\n'%mae)
        log.write('log_mae   = %+f\n'%log_mae)
        log.write('\n')




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()

    print('\nsucess!')