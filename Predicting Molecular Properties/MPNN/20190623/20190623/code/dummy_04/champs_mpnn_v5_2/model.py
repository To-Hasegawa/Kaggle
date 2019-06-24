from common import *
import torch_geometric.nn as gnn
#
# class Swish(nn.Module):
#     def __init__(self,):
#         super(Swish, self).__init__()
#     def forward(self, x):
#         sigmoid = torch.sigmoid(x)
#         return x * sigmoid

# class Scale(nn.Module):
#     def __init__(self, in_channel):
#         super(Scale, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(in_channel, dtype=torch.float32))
#         self.beta  = nn.Parameter(torch.zeros(in_channel, dtype=torch.float32))
#     def forward(self, x):
#         return self.gamma*x+self.beta
#
class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.1)
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


#message passing
class Net(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(Net, self).__init__()

        self.num_message_passing = 6
        node_hidden_dim=128
        edge_hidden_dim=128

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 64),
            nn.ReLU(),
            LinearBn(64, node_hidden_dim),
        )
        edge_net = nn.Sequential(
            LinearBn(edge_dim, 32),
            nn.ReLU(), #Swish(),#nn.ReLU(), LeakyReLU
            LinearBn(32, 64),
            nn.ReLU(), #Swish(),#nn.ReLU(),
            LinearBn(64, edge_hidden_dim),
            nn.ReLU(), #Swish(),#nn.ReLU(),
            LinearBn(edge_hidden_dim, node_hidden_dim * node_hidden_dim) # edge_hidden_dim,  node_hidden_dim *node_hidden_dim
        )

        self.conv = gnn.NNConv(node_hidden_dim, node_hidden_dim, edge_net, aggr='mean', root_weight=True) #node_hidden_dim, node_hidden_dim
        self.gru  = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = gnn.Set2Set(node_hidden_dim, processing_steps=6) # node_hidden_dim

        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(4*node_hidden_dim, 512),  #node_hidden_dim
            nn.ReLU(),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_batch_index, coupling_index, coupling_type, coupling_batch_index):

        #----
        edge_index = edge_index.t().contiguous()

        x = F.relu(self.preprocess(node))
        h = x.unsqueeze(0)

        for i in range(self.num_message_passing):
            m    = F.relu(self.conv(x, edge_index, edge))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)
        #x =  num_node, node_hidden_dim

        pool = self.set2set(x, node_batch_index) # global pool
        pool = torch.index_select(              # local select
            pool,
            dim=0,
            index=coupling_batch_index
        )
        x = torch.index_select(
            x,
            dim=0,
            index=coupling_index.view(-1)
        ).reshape(len(coupling_index),-1)

        x = torch.cat([pool,x],-1)
        predict = self.predict(x)

        predict = torch.gather(predict,1,coupling_type.view(-1,1)).view(-1)
        return predict


def criterion(predict, coupling_value):
    predict = predict.view(-1)
    coupling_value = coupling_value.view(-1)
    assert(predict.shape==coupling_value.shape)

    loss = F.mse_loss(predict, coupling_value)
    return loss



##################################################################################################################

def make_dummy_data(node_dim, edge_dim, num_target, batch_size):

    #dummy data
    num_node = []
    num_edge = []

    node = []
    edge = []
    edge_index  = []
    batch_node_index = []

    coupling_index = []
    coupling_type  = []
    coupling_value = []
    batch_coupling_index = []


    for b in range(batch_size):
        node_offset = sum(num_node)
        edge_offset = sum(num_edge)

        N = np.random.choice(10)+8
        E = np.random.choice(10)+16
        node.append(np.random.uniform(-1,1,(N,node_dim)))
        edge.append(np.random.uniform(-1,1,(E,edge_dim)))

        edge_index.append(np.random.choice(N, (E,2))+node_offset)
        batch_node_index.extend([b]*N)

        #---
        C = np.random.choice(10)+1
        coupling_index.append(np.random.choice(N,(C,2))+node_offset)
        coupling_type.append(np.random.choice(num_target, C))
        coupling_value.append(np.random.uniform(-1,1, C))
        batch_coupling_index.extend([b]*C)

        #---
        num_node.append(N)
        num_edge.append(E)





    node = torch.from_numpy(np.concatenate(node)).float().cuda()
    edge = torch.from_numpy(np.concatenate(edge)).float().cuda()
    edge_index  = torch.from_numpy(np.concatenate(edge_index)).long().cuda()
    batch_node_index = torch.from_numpy(np.array(batch_node_index)).long().cuda()


    #---
    coupling_index = torch.from_numpy(np.concatenate(coupling_index)).long().cuda()
    coupling_type  = torch.from_numpy(np.concatenate(coupling_type)).long().cuda()
    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float().cuda()
    batch_coupling_index = torch.from_numpy(np.array(batch_coupling_index)).long().cuda()


    return node,edge,edge_index, batch_node_index, coupling_index,coupling_type,coupling_value,batch_coupling_index



def run_check_net():

    #dummy data
    node_dim = 5
    edge_dim = 7
    num_target = 8
    batch_size = 16
    node,edge,edge_index, batch_node_index, coupling_index,coupling_type,coupling_value,batch_coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)

    print('batch_size ', batch_size)
    print('----')
    print('node',node.shape)
    print('edge',edge.shape)
    print('edge_index',edge_index.shape)
    print('batch_node_index',batch_node_index.shape)
    print(batch_node_index)
    print('----')

    print('coupling_index',coupling_index.shape)
    print('coupling_type',coupling_type.shape)
    print('coupling_value',coupling_value.shape)
    print('batch_coupling_index',batch_coupling_index.shape)
    print(batch_coupling_index)
    print('')

    #---
    net = Net(node_dim=node_dim, edge_dim=edge_dim, num_target=num_target).cuda()
    net = net.eval()

    predict = net(node,edge,edge_index, batch_node_index, coupling_index, coupling_type, batch_coupling_index )

    print('predict: ', predict.shape)
    print(predict)
    print('')



def run_check_train():

    node_dim = 15
    edge_dim = 5
    num_target =12
    batch_size = 64
    node,edge,edge_index, batch_node_index, coupling_index,coupling_type,coupling_value,batch_coupling_index = \
        make_dummy_data(node_dim, edge_dim, num_target, batch_size)


    net = Net(node_dim=node_dim, edge_dim=edge_dim, num_target=num_target).cuda()
    net = net.eval()


    predict = net(node,edge,edge_index, batch_node_index, coupling_index, coupling_type, batch_coupling_index )
    loss = criterion(predict, coupling_value)


    print('*loss = %0.5f'%( loss.item(),))
    print('')

    print('predict: ', predict.shape)
    print(predict)
    print(coupling_value)
    print('')

    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.01, momentum=0.9, weight_decay=0.0001)

    print('--------------------')
    print('[iter ]  loss       ')
    print('--------------------')

    i=0
    optimizer.zero_grad()
    while i<=500:
        net.train()
        optimizer.zero_grad()

        predict = net(node,edge,edge_index, batch_node_index, coupling_index,coupling_type,batch_coupling_index)
        loss = criterion(predict, coupling_value)

        loss.backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f  '%(
                i,
                loss.item(),
            ))
        i = i+1
    print('')

    #check results
    print(predict[:5])
    print(coupling_value[:5])
    print('')





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_net()
    run_check_train()

    print('\nsucess!')

