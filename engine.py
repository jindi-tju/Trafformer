# coding=utf-8
import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, args, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = wr_xformer_single(args=args, device=device, num_nodes=num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        # for param_tensor in self.model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor, '\t', self.model.state_dict()[param_tensor].size())

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.gradient_accumulation_steps = 16
        self.cur_step = 0
        self.model.train()
        self.optimizer.zero_grad()


    def train(self, input, real_val):
        self.model.train()

        # self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))  # 这个在最后1维加了填充？
        # torch.Size([64, 2, 207, 13])
        output = self.model(input, real_val)
        real_val = real_val[:, 0, :, :]
        output = output.transpose(1,3)
        # torch.Size([64, 1, 207, 12])
        # 用12个时间点预测12个时间点？ 另外原始数据好像标签也是2维的，这里怎么变成了1维
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss2 = loss/self.gradient_accumulation_steps
        loss2.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.cur_step += 1
        if self.cur_step % self.gradient_accumulation_steps == 0:  # TODO
            # print("step()")
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.cur_step = 0


        # self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input, real_val)
        real_val = real_val[:, 0, :, :]
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
