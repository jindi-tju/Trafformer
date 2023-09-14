# coding=utf-8
import torch
import numpy as np
import argparse
import time
import util
# import matplotlib.pyplot as plt
from engine import trainer
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')  # TODO cuda:0
parser.add_argument('--data',type=str,default='METR-LA',help='data path')  # METR-LA PEMS-BAY
parser.add_argument('--adjdata',type=str,default='../data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')  # TODO 100
parser.add_argument('--print_every',type=int,default=100,help='')  # TODO 100
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()


def main():
    if args.data == 'METR-LA':
        args.num_nodes = 207
        args.adjdata = '../data/sensor_graph/adj_mx.pkl'

    if args.data == 'PEMS-BAY':
        args.num_nodes = 325
        args.adjdata = '../data/sensor_graph/adj_mx_bay.pkl'
    # --gcn_bool --adjtype doubletransition --addaptadj --randomadj
    args.gcn_bool = True
    args.addaptadj = True
    args.randomadj = False
    args.in_dim = 1  # 结点流量1维，时间2维
    args.early_stopping = 10
    args.batch_size = 11  # 128

    # Xformer
    args.enc_in = args.in_dim  # args.num_nodes * args.in_dim
    args.dec_in = args.in_dim  #
    args.d_model = 32  # 32 512
    args.n_heads = 4  # 4
    args.d_ff = 256  # 256 2048
    args.moving_avg = 4 # 25 auto
    args.factor = 50  # 5
    args.activation = 'gelu'
    args.d_layers = 1
    args.e_layers = 2
    args.out_size = args.in_dim #  args.num_nodes  # TODO t_and_s
    args.dropout = 0.0  # 0.05
    args.learning_rate = 0.002  # trans 0.001 autoformer 0.00005 gwnet 0.001
    args.embed = 'timeF'
    args.freq = '10min'

    args.seq_len = args.seq_length
    args.label_len = args.seq_length
    args.pred_len = args.seq_length
    args.output_attention = False

    args.enc_attn = 'full'
    args.distil = False # True
    args.mix = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    args.data = "../data_wr_sin/"+args.data  # wr add

    print(args, flush=True)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
.
    engine = trainer(args, scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    cur_min_mvalid_loss = None
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)  # 这里只用了y的其中一维？
            # metrics = engine.train(trainx, trainy[:,0,:,:])  # x[64, 2, 207, 12], y[64, 207, 12]
            metrics = engine.train(trainx, trainy)  # x[64, 2, 207, 12], y[64, 207, 12]
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            # if iter == 1:  # TODO
            #     break
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            # metrics = engine.eval(testx, testy[:,0,:,:])
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            # if iter == 1:  # TODO
            #     break
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")


        if cur_min_mvalid_loss == None or mvalid_loss < cur_min_mvalid_loss:
            cur_min_mvalid_loss = mvalid_loss
            cur_min_epoch = i
        if i - cur_min_epoch > 0:
            print(f"early_stopping count ({(i - cur_min_epoch)}/{args.early_stopping})")
            # if i - cur_min_epoch > 4:
            #     args.learning_rate = args.learning_rate/2
            #     print(f"change learing_rate to{args.learning_rate}")
                # engine.optimizer.
        if i - cur_min_epoch >= args.early_stopping:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)), flush=True)  # TODO add flush


    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    engine.model.eval()  # TODO 加的
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, testy).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)), flush=True)


    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:pred.shape[0], :pred.shape[1], i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
