import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import os
import pandas as pd
import argparse
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

DATABASE_DIR = './database/Data.db'

NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365




parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--total_step', type=int, default=80000)
parser.add_argument('--x_window_size', type=int, default=31)
#parser.add_argument('--y_window_size', type=int, default=11)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--coin_num', type=int, default=11)
parser.add_argument('--feature_number', type=int, default=4)
parser.add_argument('--output_step', type=int, default=500)
parser.add_argument('--model_index', type=int, default=0)
parser.add_argument('--multihead_num', type=int, default=2)
parser.add_argument('--local_context_length', type=int, default=3)
parser.add_argument('--model_dim', type=int, default=12)
parser.add_argument('--test_portion', type=float, default=0.08)
parser.add_argument('--trading_consumption', type=float, default=0.0025)
parser.add_argument('--variance_penalty', type=float, default=0.0)
parser.add_argument('--cost_penalty', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=5e-9)
parser.add_argument('--daily_interest_rate', type=float, default=0.001)
parser.add_argument('--start', type=str, default = "2016/01/01")
parser.add_argument('--end', type=str, default = "2018/01/01")
parser.add_argument('--model_name', type=str, default = None)
parser.add_argument('--log_dir', type=str, default = None)
parser.add_argument('--model_dir', type=str, default = None)

FLAGS = parser.parse_args()



def parse_time(time_string):
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())






class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, coin_number, end, volume_average_days=1, volume_forward=0, online=True):
        self.initialize_db()
        self.__storage_period = DAY #FIVE_MINUTES  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        #if self._online:
        #    self._coin_list = CoinList(end, volume_average_days, volume_forward)  # 这一步获取需要下载的coin列表
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(start, end, period, features).values


    def get_global_panel(self, start, end, period=300, features=('close',)):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start = int(start - (start%period)) 
        end = int(end - (end%period))  
        
        # 加载数据/opt/ml/disk/database
#        df = pd.read_csv('/data2/kaylakxu/PGPortfolio-master/PGPortfolio-master/mtcn_SP500/mtcn_SP500/database/all_stocks_5yr.csv')
        df = pd.read_csv('./database/all_stocks_5yr.csv')
        df.describe()
        df.head(5)
        
        # asset name list
        name_list = df['Name']
        coins = name_list.drop_duplicates(keep='first',inplace=False).tolist()
        self.__coins = coins
        #name_list1= name_list.drop_duplicates(keep='first',inplace=True)
        #coins = name_list.tolist()
        
        #coins = self.select_coins(start=end - self.__volume_forward - self.__volume_average_days * DAY,
         #                         end=end-self.__volume_forward) # 挑选coins
        
        time = df['date']
        time_index=time.drop_duplicates(keep='first',inplace=False).tolist()
        #for coin in coins:
        #    self.update_data(start, end, coin)  # 更新数据

        if len(coins)!=self._coin_number:
            raise ValueError("the length of selected coin (%d) is not equal to expected %d"
                             % (len(coins), self._coin_number))

        print("feature type list is %s" % str(features))
        self.__checkperiod(period)  
        features=['open','high','low','close']
        panel = pd.Panel(items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32)

        if os.path.exists('./database/data_new1.pkl'):
            print('read_data')
            panel = pd.read_pickle("./database/data_new1.pkl")
        else:
            print('process data')
            for sample_number, coin in enumerate(coins):
                for feature in features:
                    temp_data = df.loc[df["Name"]==coin]
                    temp_data = temp_data.set_index('date')
                    temp_data =temp_data[feature]
                    if np.sum(np.isnan(temp_data))>0:
                        temp_data =temp_data.fillna(method='backfill')
                    temp=0
                    if np.shape(temp_data)[0]!= np.shape(time_index)[0]:
                        temp_data1 = pd.DataFrame(np.random.randn(1259),index=time_index)
                        for time_id in time_index:
                            if time_id in temp_data.index.tolist():
                                temp = temp_data.loc[time_id]
                                temp_data1.loc[time_id] = temp
                            else:
                                if time_id == '2013-02-08':
                                    temp = 1
                                    temp_data1.loc[time_id] = temp
                                else:
                                    temp_data1.loc[time_id] = temp
                        temp_data = temp_data1.squeeze()
                    else:
                        temp_data.index = time_index
                    panel.loc[feature, coin, time_index] = temp_data.squeeze()
            f = open('./database/data_new1.pkl','wb')
            panel.to_pickle(f)
            print('save_data')
            f.close            
        return panel



    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')






def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list

def get_volume_forward(time_span, portion, portion_reversed):
    """
    :param portion_reversed: if False, the order to sets are [train, validation, test]
    else the order is [test, validation, train]
    """
    volume_forward = 0
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward

def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    frames = {}
    for item in panel.items:
        if type == "both":
            frames[item] = panel.loc[item].fillna(axis=1, method="bfill").\
                fillna(axis=1, method="ffill")
        else:
            frames[item] = panel.loc[item].fillna(axis=1, method=type)
    return pd.Panel(frames)


class DataMatrices:
    def __init__(self, start, end, period, batch_size=50, volume_average_days=30, buffer_bias_ratio=0,
                 market="poloniex", coin_filter=1, window_size=50, feature_number=3, test_portion=0.15,
                 portion_reversed=False, online=False, is_permed=False):
        """
        :param start: Unix time
        :param end: Unix time
        :param period: the length of the period. For example, each period has a length of 30 minutes, then period=1800
        :param coin_filter: number of coins that would be selected
        :param window_size: periods of input data
        :param feature_number: an int indicates the number of features
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        :param is_permed: if False, the sample inside a mini-batch is in order
        """
        start = int(start)
        self.__end = int(end)

        # assert window_size >= MIN_NUM_PERIOD
        self.__coin_no = coin_filter
        type_list = get_type_list(feature_number)
        self.__features = type_list
        self.feature_number = feature_number
        volume_forward = get_volume_forward(self.__end-start, test_portion, portion_reversed)
        self.__history_manager = HistoryManager(coin_number=coin_filter, end=self.__end,
                                                    volume_average_days=volume_average_days,
                                                    volume_forward=volume_forward, online=online)
        # 通过historyManager这一函数，初始化要收集的数据
        self.__global_data = self.__history_manager.get_global_panel(start, self.__end, period=period, features=type_list)
        self.__period_length = period
        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=self.__global_data.minor_axis, columns=self.__global_data.major_axis)
        self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)

        self._window_size = window_size
        self._num_periods = len(self.__global_data.minor_axis)
        self.__divide_data(test_portion, portion_reversed)

        self._portion_reversed = portion_reversed
        self.__is_permed = is_permed

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased
        end_index = self._train_ind[-1]
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                               end_index=end_index,
                                               sample_bias=buffer_bias_ratio,
                                               batch_size=self.__batch_size,
                                               coin_number=self.__coin_no,
                                               is_permed=self.__is_permed)

        print("the number of training examples is %s"
                     ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        print("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        print("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))

    @property
    def global_weights(self):
        return self.__PVM

    @staticmethod
    def create_from_config(config):
        """main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        return DataMatrices(start=start,
                            end=end,
                            market=input_config["market"],
                            feature_number=input_config["feature_number"],
                            window_size=input_config["window_size"],
                            online=input_config["online"],
                            period=input_config["global_period"],
                            coin_filter=input_config["coin_number"],
                            is_permed=input_config["is_permed"],
                            buffer_bias_ratio=train_config["buffer_biased"],
                            batch_size=train_config["batch_size"],
                            volume_average_days=input_config["volume_average_days"],
                            test_portion=input_config["test_portion"],
                            portion_reversed=input_config["portion_reversed"],
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.__history_manager.coins

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def test_indices(self):
        return self._test_ind[:-(self._window_size+1):]

    @property
    def num_test_samples(self):
        return self._num_test_samples

    def append_experience(self, online_w=None):
        """
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        self._train_ind.append(self._train_ind[-1]+1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def get_test_set(self):
        return self.__pack_samples(self.test_indices)
    
    def get_test_set_online(self,ind_start,ind_end, x_window_size):
        return self.__pack_samples_test_online(ind_start,ind_end, x_window_size)
    
    def get_training_set(self):
        return self.__pack_samples(self._train_ind[:-self._window_size])
##############################################################################
    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
#        print(np.shape([exp.state_index for exp in self.__replay_buffer.next_experience_batch()]),[exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs-1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w
#            print("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}
    
    def __pack_samples_test_online(self, ind_start, ind_end, x_window_size):
#        indexs = np.array(indexs)
        last_w = self.__PVM.values[ind_start-1:ind_start, :]
#        y_window_size = window_size-x_window_size
        def setw(w):
            self.__PVM.iloc[ind_start, :] = w
#            print("set w index from %d-%d!" %( indexs[0],indexs[-1]))
        M = [self.get_submatrix_test_online(ind_start,ind_end)]  #[1,4,11,2807]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, x_window_size:]/ M[:, 0, None, :, x_window_size-1:-1]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}
##############################################################################################    
    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind:ind+self._window_size+1]
    
    def get_submatrix_test_online(self, ind_start,ind_end):
        return self.__global_data.values[:, :, ind_start:ind_end]
    
    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self.test_indices)


class ReplayBuffer:
    def __init__(self, start_index, end_index, batch_size, is_permed, coin_number, sample_bias=1.0):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__coin_number = coin_number
        self.__experiences = [Experience(i) for i in range(start_index, end_index)]
        self.__is_permed = is_permed
        # NOTE: in order to achieve the previous w feature
        self.__batch_size = batch_size
        self.__sample_bias = sample_bias
        print("buffer_bias is %f" % sample_bias)

    def append_experience(self, state_index):
        self.__experiences.append(Experience(state_index))
        print("a new experience, indexed by %d, was appended" % state_index)

    def __sample(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
        if self.__is_permed:
            for i in range(self.__batch_size):
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,
                                        self.__sample_bias)
            batch = self.__experiences[batch_start:batch_start+self.__batch_size]
        return batch
class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, batch_size, coin_num, window_size, feature_number,
                 d_model_Encoder,d_model_Decoder, encoder, decoder, src_pe, tgt_pe, local_context_length):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size=batch_size
        self.coin_num=coin_num
        self.window_size=window_size
        self.feature_number=feature_number
        self.d_model_Encoder=d_model_Encoder
        self.d_model_Decoder=d_model_Decoder
        self.linear_src= nn.Linear(in_features=feature_number,out_features=d_model_Encoder)
        self.linear_tgt= nn.Linear(in_features=feature_number,out_features=d_model_Decoder)
#        self.linear_src_2_embedding=nn.Linear(in_features=d_model_Encoder,out_features=d_model_Decoder)
        self.src_pe = src_pe
        self.tgt_pe = tgt_pe
        self.local_context_length = local_context_length
        self.linear_out=nn.Linear(in_features=1+d_model_Encoder,out_features=1)
        self.linear_out2=nn.Linear(in_features=1+d_model_Encoder,out_features=1)

        self.bias = torch.nn.Parameter(torch.zeros([1,1,1]))
        self.bias2 = torch.nn.Parameter(torch.zeros([1,1,1]))
    def forward(self, src, tgt, previous_w, src_mask, tgt_mask, padding_price):
        #(3,109,31,11)
        src=src/src[0:1,:,-1:,:]
        src=src.permute(3,1,2,0) #(3,109,31,11)->(11,109,31,3)
#        print("src",src.size())
        src=src.contiguous().view(src.size()[0]*src.size()[1],self.window_size,self.feature_number) #(11,109,31,3)->(11*109,31,3)
#        print("src",src.size())
        src=self.linear_src(src)                   #(11*109,31,3)->(11*109,31,512)
        src=self.src_pe(src)                       #(11*109,31,512) 
#        src=src.view(self.feature_number,-1,self.window_size,self.d_model_Encoder)  ##(3*109,31,512)->#(3,109,31,512)
        src=src.view(self.coin_num,-1,self.window_size,self.d_model_Encoder)  ##(11*109,31,512)->#(11,109,31,512)
#        print("src_mask:",src_mask.size())
#        print("src:",src.size())
        encode_out=self.encoder(src, src_mask)
#        encode_out=self.linear_src_2_embedding(encode_out)
#        print("encode_out:",encode_out) 
###########################padding price#######################################################################################
        if(padding_price is not None):
            tgt=torch.cat([padding_price,tgt],2)    #(11,109,1,3) cat (11,109,2+2,3) ->#(11,109,5,3)
            tgt=tgt.contiguous().view(tgt.size()[0]*src.size()[1],self.local_context_length*2-1,self.feature_number)  #(11,109,5,3)->(11*109,5,3) 
        else:
            tgt=tgt.contiguous().view(tgt.size()[0]*src.size()[1],1,self.feature_number)
##############对current price除以最后一日收盘价################################
        tgt=tgt/tgt[:,-1:,0:1]
########################################################################################
        tgt=self.linear_tgt(tgt)                   #[64, 1, 16]->[64, 1, 512]    ##(11*109,3,3)->(11*109,3,512)
        tgt=self.tgt_pe(tgt)  #[11*109,5,512]
#        pre_w=pre_w.repeat(self.coin_num,1,1,1)   ##[11*109,1,512]
        if(padding_price is not None):
            padding_price=tgt[:,:-self.local_context_length,:]               #留出来3个时刻给self-attetion
            padding_price=padding_price.view(self.coin_num,-1,self.local_context_length-1,self.d_model_Decoder)   #[11,109,2,512]

#pre_w：[11*109,3,512]->[11,109,3,512]
        tgt=tgt[:,-self.local_context_length:,:]    #[11*109,3,512]
#        print("pre_w:",pre_w.size())
        tgt=tgt.view(self.coin_num,-1,self.local_context_length,self.d_model_Decoder)  
#################################padding_price=None###########################################################################
        decode_out=self.decoder(tgt, encode_out, src_mask, tgt_mask, padding_price)
#        print("decode_out:",decode_out.size()) #[11, 109, 1, 512]
        decode_out=decode_out.transpose(1,0)       #[11, 109, 1, 512]->#[109, 11, 1, 512]   
        decode_out=torch.squeeze(decode_out,2)     #[109, 11, 1, 512]->#[109, 11, 512]

        previous_w=previous_w.permute(0,2,1)  #[109,1,11]->109,11,1]
        out=torch.cat([decode_out,previous_w],2)
#            out=decode_out
        out2=self.linear_out2(out)
        out=self.linear_out(out)  #[109, 11, 512+11]->[109, 11, 1]

        bias=self.bias.repeat(out.size()[0],1,1)   #
        bias2=self.bias2.repeat(out2.size()[0],1,1)

        out=torch.cat([bias,out],1) #[109,12,1]
        out2=torch.cat([bias2,out2],1)

        out=out.permute(0,2,1) #[109,1,12]
        out2=out2.permute(0,2,1)

        out=F.softmax(out, dim = -1)
        out2=F.softmax(out2, dim = -1)

        out=out*2
        out2=-out2
        return out+out2 #[109,1,12]



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):    #[64,10,512]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
#        print("LayerNorm",(self.a_2 * (x - mean) / (std + self.eps) + self.b_2).size())   [64,10,512]
#        print("LayerNorm",self.a_2.size())   
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
#            print("Encoder:",x)
            x = layer(x, mask)
#            print("Encoder:",x.size())
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout=dropout
        if(dropout!=0):
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if (self.dropout!=0):
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + sublayer(self.norm(x))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, None, None))
        return self.sublayer[1](x, self.feed_forward)

######################################Decoder############################################
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask, padding_price):
#        print("x:",np.shape(x))                  #x: torch.Size([64, 1, 512])   tgt_y
#        print("memory:",np.shape(memory))      #memory: torch.Size([64, 10, 512])   encode_out
#        print("src_mask:",np.shape(src_mask))   #src_mask: torch.Size([64, 1, 10])
#        print("tgt_mask:",np.shape(tgt_mask))   #tgt_mask: torch.Size([64, 1, 1])
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, padding_price)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask, padding_price):  
#x: torch.Size([30, 9, 512])   tgt_embed   
#memory: torch.Size([30, 10, 512])   encode_out
#src_mask: torch.Size([30, 1, 10])
#tgt_mask: torch.Size([30, 9, 9])
        "Follow Figure 1 (right) for connections."
        m = memory 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, padding_price, padding_price))  # padding_price_q,padding_price_k
#        print("x0 shape:",np.shape(x))  # x0 shape: torch.Size([30, 9, 512])
        #[11,109,31,512]->[11,109,1,512]
        x = x[:,:,-1:,:]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, None, None))
#        print("x1 shape:",np.shape(x))  # x1 shape: torch.Size([30, 1, 512])
        return self.sublayer[2](x, self.feed_forward)



def subsequent_mask(size):   #size tgt 时序维度 1
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')         #上三角为1
#    print(subsequent_mask)
#    print(torch.from_numpy(subsequent_mask))
    return torch.from_numpy(subsequent_mask) == 0   #翻轉為下三角
#    return (subsequent_mask == 0).astype('uint8')   #翻轉為下三角

def tf_subsequent_mask(size):   #size tgt 时序维度 1
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')         #上三角为1
#    print(subsequent_mask)
#    print(torch.from_numpy(subsequent_mask))
#    return torch.from_numpy(subsequent_mask) == 0   #翻轉為下三角
    return (subsequent_mask == 0).astype('uint8')   #翻轉為下三角


def attention(query, key, value, sigmoid=True, mask=None, dropout=None):
#decoder:query, key, value:[30,8,9,64]
#mask:[30, 9, 9]
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) #64
    if(sigmoid):
        l1_q= query.unsqueeze(3)     #[30,8,9,64]->[30,8,9,1,64]
#    print("l1_q",l1_q.size())
        l1_k= key.unsqueeze(2)       #[30,8,9,64]->[30,8,1,9,64]
#    print("l1_k",l1_k.size())
        l1=torch.abs(l1_q-l1_k)              #[30,8,9,9,64]
#    print("l1",l1.size())
        l1=torch.sum(l1,-1)             #[30,8,9,9]
#    print("l1",l1.size())
        sigmoid_l1=2*torch.sigmoid(-l1)    #[30,8,9,9]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
#    print("scores:",np.shape(scores))     #[30, 8, 9, 9] 
#    print("query:",np.shape(query))    
#    print("key.transpose(-2, -1):",np.shape(key.transpose(-2, -1)))    #key.transpose(-2, -1): torch.Size([30, 8, 64, 9])
    if mask is not None:
#        print("mask == 0",(mask == 0).size(),mask == 0)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)  #[30, 8, 9, 9] 
    if(sigmoid):
        p_attn = p_attn*sigmoid_l1
#    print("p_attn:",np.shape(p_attn))   #[64, 8, 10, 10]
#    print("value:",np.shape(value))     #[64, 8, 10, 64]
    if dropout is not None:
        p_attn = dropout(p_attn)
#    print("torch.matmul(p_attn, value):", np.shape(torch.matmul(p_attn, value)))   #[64, 8, 10, 64]
    return torch.matmul(p_attn, value), p_attn

def feature_attention(value, weight):   #[64, 8, 10, 64]
    value = value.transpose(1, 2) #[64, 10, 8, 64]
#    print("value1:",np.shape(value))
    value = weight.mul(value)    #[64,10,8,64] * [64, 10, 8, 64]  点乘
#    print("value2:",np.shape(value))
    value = value.transpose(1, 2) #[64, 8, 10, 64]
#    print("value3:",np.shape(value))
    return value
#value1: torch.Size([64, 10, 8, 64])
#value2: torch.Size([64, 10, 8, 64])
#value3: torch.Size([64, 8, 10, 64])


class MultiHeadedAttention(nn.Module):
    def __init__(self, asset_atten, h, d_model, dropout, local_context_length):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.local_context_length=local_context_length
        self.linears = clones(nn.Linear(d_model, d_model), 2)
#        self.conv_q= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
#        self.conv_k= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
        self.ass_linears_v = nn.Linear(d_model, d_model)
        self.ass_conv_q= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
        self.ass_conv_k= nn.Conv2d(d_model, d_model, (1,1), stride=1, padding=0, bias=True)
        
        self.attn = None
        self.attn_asset = None
        self.dropout = nn.Dropout(p=dropout)
        self.feature_weight_linear=nn.Linear(d_model, d_model)
        self.asset_atten=asset_atten
#        print("x0",x.size()) #[109, 3, 11, 30] 对
#        x=self.conv2(x)  #[109,10,11,1] 对
        
    def forward(self, query, key, value, mask, padding_price_q,padding_price_k):
#decode：query, key, value ：       [64, 1, 512]  (3*109,31,512)  q [3,109,1,512] or (3,109,31,512) k,v(3,109,31,512)
#decode：mask：[64, 1, 1]                                         q (11,109,1,512) or (11,109,31,512)
        "Implements Figure 2"
#        print("mask shape:",mask.size())
        
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)       #  [109,1,1,31] [109,1,1,1]
            mask = mask.repeat(query.size()[0], 1, 1, 1)  # [109*3,1,1,31]  [109*3,1,1,1]    #[9, 1, 1, 31]
            mask = mask.cuda()                            # 
#            print("mask shape:",mask.size())
#        print("mask:",mask.size())
        q_size0 = query.size(0)  #11
        q_size1 = query.size(1)  #109
        q_size2 = query.size(2)  #31 0r 1
        q_size3 = query.size(3)  #512
        key_size0 = key.size(0)  
        key_size1 = key.size(1)  
        key_size2 = key.size(2)  
        key_size3 = key.size(3)  
##################################query#################################################    
        if(padding_price_q is not None):
            padding_price_q =padding_price_q.permute((1,3,0,2))   #[11,109,2,512]->#[109,512,11,2]
            padding_q=padding_price_q
        else:
            if(self.local_context_length>1):
                padding_q = torch.zeros((q_size1,q_size3,q_size0,self.local_context_length-1)).cuda()
#                print("padding_q is 0 \r\n")
            else:
                padding_q = None
#                print("padding_q is None \r\n")
        query = query.permute((1,3,0,2))
#        print("type(padding_q)",type(padding_q),"\r\n")
        if(padding_q is not None):
#            print("padding_q",padding_q.size(),"query",query.size())
            query = torch.cat([padding_q,query],-1)
#######################################################################################################################
#        query = self.conv_q(query)   #[109,512,11,31+2] conv -> [109,512,11,31] 
##########################################local-attention############################################################################
        #linar
#        query = self.conv_q(query)
        #[109,512,11,31+2]->[109,11,31+2,512]
        query=query.permute((0,2,3,1))
        #[109,11,31+2-2,512] *[109,11,512,31+2]->[109,11,31,31+2]
        local_weight_q=torch.matmul(query[:,:,self.local_context_length-1:,:], query.transpose(-2, -1))/ math.sqrt(q_size3)
        #[109,11,31,31+2]->[109,11,1,3*31]
        local_weight_q_list=[F.softmax(local_weight_q[:,:,i:i+1,i:i+self.local_context_length], dim = -1) for i in range(q_size2)]
        local_weight_q_list=torch.cat(local_weight_q_list,3)
        #[109,11,1,3*31]->[109,11,3*31,1]
        local_weight_q_list=local_weight_q_list.permute(0,1,3,2)
        #[109,11,31+2,512]->[109,11,3*31,512]
        q_list=[query[:,:,i:i+self.local_context_length,:] for i in range(q_size2)]
        q_list=torch.cat(q_list,2)
        #[109,11,3*31,1]*[109,11,3*31,512]->[109,11,3*31,512]
        query=local_weight_q_list*q_list
        #>[109,11,3*31,512]->[109,11,3,31,512]
        query=query.contiguous().view(q_size1,q_size0,self.local_context_length,q_size2,q_size3)
        #[109,11,3,31,512]->[109,11,31,512]
        query=torch.sum(query,2)
        #[109,11,31,512]->[109,512,11,31]
        query=query.permute((0,3,1,2))
######################################################################################
        query = query.permute((2,0,3,1))    #[109,512,11,31] ->[11,109,31,512] 
#        print("query:",query.size())  #[11, 128, 31, 22]
        query = query.contiguous().view(q_size0*q_size1,q_size2,q_size3) #[11,109,31,512] -> [11*109,31,512] 
        query = query.contiguous().view(q_size0*q_size1,q_size2,self.h, self.d_k).transpose(1, 2) 
        #[11*109,31,512] ->[11*109,31,8,64]->[11*109,8,31,64] 
#        print("query:",query.size())
#####################################key#################################################
        if(padding_price_k is not None):
            padding_price_k =padding_price_k.permute((1,3,0,2))   #[11,109,2,512]->#[109,512,11,2]
            padding_k=padding_price_k
        else:
            if(self.local_context_length>1):
                padding_k = torch.zeros((key_size1,key_size3,key_size0,self.local_context_length-1)).cuda()
            else:
                padding_k = None
        key = key.permute((1,3,0,2)) 
        if(padding_k is not None):
            key = torch.cat([padding_k,key],-1) 
##########################################local-attention############################################################################
        #[30,8,9,64]   [109,1,31,512]
#        key=self.conv_k(key)
        #[109,512,11,31+2]->[109,11,31+2,512]
        key=key.permute((0,2,3,1))
        #[109,11,31+2-2,512] *[109,11,512,31+2]->[109,11,31,31+2]
        local_weight_k=torch.matmul(key[:,:,self.local_context_length-1:,:], key.transpose(-2, -1))/ math.sqrt(key_size3)
        #[109,11,31,31+2]->[109,11,1,3*31]
        local_weight_k_list=[F.softmax(local_weight_k[:,:,i:i+1,i:i+self.local_context_length], dim = -1) for i in range(key_size2)]
        local_weight_k_list=torch.cat(local_weight_k_list,3)
        #[109,11,1,3*31]->[109,11,3*31,1]
        local_weight_k_list=local_weight_k_list.permute(0,1,3,2)
        #[109,11,31+2,512]->[109,11,3*31,512]
        k_list=[key[:,:,i:i+self.local_context_length,:] for i in range(key_size2)]
        k_list=torch.cat(k_list,2)
        #[109,11,3*31,1]*[109,11,3*31,512]->[109,11,3*31,512]
        key=local_weight_k_list*k_list
        #>[109,11,3*31,512]->[109,11,3,31,512]
        key=key.contiguous().view(key_size1,key_size0,self.local_context_length,key_size2,key_size3)
        #[109,11,3,31,512]->[109,11,31,512]
        key=torch.sum(key,2)
        #[109,11,31,512]->[109,512,11,31]
        key=key.permute((0,3,1,2))
#########################################################################################################################
#        key = self.conv_k(key)
#########################################################################################################################
        key = key.permute((2,0,3,1))
        key = key.contiguous().view(key_size0*key_size1,key_size2,key_size3)
        key = key.contiguous().view(key_size0*key_size1,key_size2,self.h, self.d_k).transpose(1, 2) 
#####################################value##############################################        
        value=value.view(key_size0*key_size1,key_size2,key_size3) #(3,109,31,512)->(3*109,31,512)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        nbatches=q_size0*q_size1
        value=self.linears[0](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  #[11*109,31,8,64]
        x, self.attn = attention(query, key, value, False, mask=None, 
                                 dropout=self.dropout)
#        print("weight_value:",np.shape(x))   #weight_value: torch.Size([30, 8, 9, 64])   多头atten后的value

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
#        print("weight_value2:",np.shape(x))    #weight_value2: torch.Size([30, 9, 512])   #D[3*109,1,512] or E[3*109,31,512]
        x=x.view(q_size0,q_size1,q_size2,q_size3)    #D[11,109,1,512] or E[11,109,31,512]
#        print("self.attn ",self.attn.size())
#        print("x:",x.size())
        if(self.asset_atten):
#######################################ass_query#####################################################################    
            ass_query = x.permute((2,1,0,3))  ##D[11,109,1,512]->[1,109,11,512] or E[11,109,31,512]->[31,109,11,512]
            ass_query = ass_query.contiguous().view(q_size2*q_size1,q_size0,q_size3) #[31,109,11,512] -> [31*109,11,512] 
            ass_query = ass_query.contiguous().view(q_size2*q_size1,q_size0,self.h, self.d_k).transpose(1, 2)    #[31*109,8,11,64]     
#        print("ass_query:",ass_query.size())   #[3968, 2, 11, 11]
########################################ass_key####################################################################
            ass_key = x.permute((2,1,0,3))  ##D[11,109,1,512]->[1,109,11,512] or E[11,109,31,512]->[31,109,11,512]
            ass_key = ass_key.contiguous().view(q_size2*q_size1,q_size0,q_size3) #[31,109,11,512] -> [31*109,11,512] 
            ass_key = ass_key.contiguous().view(q_size2*q_size1,q_size0,self.h, self.d_k).transpose(1, 2)    #[31*109,8,11,64]  
 #       print("ass_key:",ass_key.size())
####################################################################################################################
            ass_value = x.permute((2,1,0,3))  ##D[11,109,1,512]->[1,109,11,512] or E[11,109,31,512]->[31,109,11,512]
            ass_value = ass_value.contiguous().view(q_size2*q_size1,q_size0,q_size3) #[31,109,11,512]->[31*109,11,512]
            ass_value = ass_value.contiguous().view(q_size2*q_size1,-1, self.h, self.d_k).transpose(1, 2)  #[31*109,8,11,64]
#        print("ass_value:",ass_value.size())
######################################################################################################################    
            x, self.attn_asset = attention(ass_query, ass_key, ass_value, False, mask=None, 
                                 dropout=self.dropout)   
            x = x.transpose(1, 2).contiguous().view(q_size2*q_size1, -1, self.h * self.d_k)  #[31*109,11,512]
            x=x.view(q_size2,q_size1,q_size0,q_size3) #[31,109,11,512]
            x=x.permute(2,1,0,3)  #[11,109,31,512]
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
#        print("ffn:",x.size())
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, start_indx, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.start_indx=start_indx
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, self.start_indx:self.start_indx+x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)




class NoamOpt:
    "Optim wrapper that implements rate."
    #512, 1, 400
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1,beta=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate


    def forward(self, w, y):  #w:[109,1,12]   y:[109,11,3] 取 close prise [109,11,1]
        close_price = y[:,:,0:1].cuda()
        #future close prise (including cash)
        close_price = torch.cat([torch.ones(close_price.size()[0],1,1).cuda(),close_price],1).cuda()         #[109,11,1]+[109,1,1]->[109,12,1]
        reward = torch.matmul(w,close_price)                                                                 #[109,1,1]
        close_price = close_price.view(close_price.size()[0],close_price.size()[2],close_price.size()[1])    #[109,1,12] 
###############################################################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(),dtype=torch.float).cuda()
        interest[element_reward<0] = element_reward[element_reward<0]
#        print("interest:",interest.size(),interest,'\r\n')
        interest = torch.sum(interest,2).unsqueeze(2)*self.interest_rate  #[109,1,1]
###############################################################################################################
        future_omega = w*close_price/reward          #[109,1,12]
        wt=future_omega[:-1]                         #[108,1,12]
        wt1=w[1:]                                    #[108,1,12]
###############################################################################################################
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio   #[108,1]
        pure_pc=pure_pc.cuda()
        pure_pc=torch.cat([torch.ones([1,1]).cuda(),pure_pc],0)
        pure_pc=pure_pc.view(pure_pc.size()[0],1,pure_pc.size()[1])  #[109,1,1]
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)
#        print("pure_pc",pure_pc.size()) 
##################Deduct transaction fee##################
        reward=reward*pure_pc    #reward=pv_vector
##################Deduct loan interest##################
        reward=reward+interest
##############################################
        portfolio_value=torch.prod(reward,0)
#        print("portfolio_value",portfolio_value.size())   #[1,1]
        batch_loss=-torch.log(reward)
#        batch_loss=-reward*logp
#        print("reward:",reward.size())
#####################variance_penalty##############################
        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean() + self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value[0][0]
        else:
            loss = batch_loss.mean()+self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value[0][0]

class SimpleLossCompute2:
    "A simple loss compute and train function."
#    def __init__(self, generator,generator_sigma, criterion, opt=None):
    def __init__(self,  criterion, opt=None):
#        self.generator = generator
#        self.generator_sigma = generator_sigma
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        loss, portfolio_value= self.criterion(x,y)
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value



def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)


class Test_Loss(nn.Module):
    def __init__(self, commission_ratio,interest_rate,gamma=0.1,beta=0.1, size_average=True):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate

    def forward(self, w, y):  #w:[109,1,12]   y:[109,11,3] 取 close prise [109,11,1]  #w:[128,10,1,12] y(128,10,11,4)
#        close_price=y[:,:,0:1].cuda()
        close_price = y[:,:,:,0:1].cuda()   #(128,10,11,1)
        #future close prise (including cash)
        close_price = torch.cat([torch.ones(close_price.size()[0],close_price.size()[1],1,1).cuda(),close_price],2).cuda()           #[128,10,11,1] cat [128,10,1,1]->[128,10,12,1]
        reward = torch.matmul(w,close_price)                                                                                         #[128,10,1,12] * [128,10,12,1] ->[128,10,1,1]
        close_price = close_price.view(close_price.size()[0],close_price.size()[1],close_price.size()[3],close_price.size()[2])      #[128,10,12,1] -> [128,10,1,12]
##############################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(),dtype=torch.float).cuda()
        interest[element_reward<0] = element_reward[element_reward<0]
        interest=torch.sum(interest,3).unsqueeze(3)*self.interest_rate                                                               #[128,10,1,1]
##############################################################################
        future_omega = w*close_price/reward                                                                                            #[128,10,1,12] * [128,10,1,12] / [128,10,1,1]
        wt = future_omega[:,:-1]                                                                                                       #[128, 9,1,12]
        wt1 = w[:,1:]                                                                                                                  #[128, 9,1,12]
        pure_pc = 1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio   #[108,1]  #[128,9,1]
        pure_pc = pure_pc.cuda()
        pure_pc = torch.cat([torch.ones([pure_pc.size()[0],1,1]).cuda(),pure_pc],1)                                                    #[128,1,1] cat [128,9,1] ->[128,10,1]
        pure_pc = pure_pc.view(pure_pc.size()[0],pure_pc.size()[1],1,pure_pc.size()[2])                                                #[128,10,1]->[128,10,1,1]
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)   #[127,  1]    [128, 9, 1]   
##################Deduct transaction fee#####################################
        reward=reward*pure_pc    #reward=pv_vector [109,1,1]*[109,1,1]      [128,10,1,1]*[128,10,1,1]  test: [1,2808-31,1,1]
##################Deduct loan interest#######################################
        reward= reward+interest
        if not self.size_average:
            tst_pc_array=reward.squeeze()
            sr_reward=tst_pc_array-1
            SR=sr_reward.mean()/sr_reward.std()
            SN=torch.prod(reward,1) #[1,1,1,1]
            SN=SN.squeeze() #
            St_v=[]
            St=1.
            MDD=max_drawdown(tst_pc_array)
            for k in range(reward.size()[1]):  #2808-31
                St*=reward[0,k,0,0]
                St_v.append(St.item())
            if(MDD==0):
                CR=0
            else:
                print("MDD:",MDD.size(),MDD)
                CR=SN/MDD
            TO=cost_penalty.mean()
############################################################################
        portfolio_value=torch.prod(reward,1)                                     #[128,1,1]
        batch_loss=-torch.log(portfolio_value)                                   #[128,1,1]
#####################variance_penalty#######################################
        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean() + self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean()+self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value[0][0][0],SR,CR,St_v,tst_pc_array,TO


class SimpleLossCompute2_tst:
    "A simple loss compute and train function."
#    def __init__(self, generator,generator_sigma, criterion, opt=None):
    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value= self.criterion(x,y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO = self.criterion(x,y)     
            return loss, portfolio_value,SR,CR,St_v,tst_pc_array,TO   



def make_std_mask(tgt,batch_size):
    "Create a mask to hide padding and future words."
    tgt_mask = (torch.ones(batch_size,1,1)==1)             #[30,9]不为0的为真,然后增加一维度[30,1,9]  时序维度，或者说是word中非padding的部分
    tgt_mask = tgt_mask & (subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data))   #[64,1,1]&[1,1,1]下三角为真 ——> [64,1,1]
    return tgt_mask    #[64, 64, 64]  [64,1,1] 全1



def train_one_step(DM,x_window_size,model,loss_compute,local_context_length):
    batch=DM.next_batch()
    batch_input = batch["X"]                                                 #[109,3,11,31]
    batch_y = batch["y"]                                                     #[109, 3, 11]
    batch_last_w = batch["last_w"]
    batch_w = batch["setw"]     

    previous_w = torch.tensor(batch_last_w, dtype=torch.float).cuda()
    previous_w = torch.unsqueeze(previous_w, 1)                              #[109, 11]->[109,1,11] 
    batch_input = batch_input.transpose((1,0,2,3))                           #[109,3,11,31]->[3,109,11,31]
    batch_input = batch_input.transpose((0,1,3,2))                           #[3,109,11,31]->[3,109,31,11]
    src = torch.tensor(batch_input,dtype = torch.float).cuda() 

    src_mask = None
    currt_price = src.permute((3,1,2,0))                                     #[3,109,31,11]->[11,109,31,3]
    if(local_context_length>1):
        padding_price = currt_price[:,:,-(local_context_length)*2+1:-1,:]                #[11,109,2,3]
    else:
        padding_price = None
    currt_price = currt_price[:,:,-1:,:]                                     #[11,109,31,3]->[11,109,1, 3]

    trg_mask = None
    batch_y = batch_y.transpose((0,2,1))                                     #[109, 3, 11]->[109,11,3]
    trg_y = torch.tensor(batch_y,dtype = torch.float).cuda()
    out = model.forward(src, currt_price, previous_w, src_mask, trg_mask, padding_price)

    new_w = out[:,:,1:]                                                      #remove cash
    new_w = new_w[:,0,:]                                                     #[109,1,11]->[109,11]
    new_w = new_w.detach().cpu().numpy()
    batch_w(new_w)  
    loss, portfolio_value = loss_compute(out,trg_y)
    return loss, portfolio_value


def test_online(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch = DM.get_test_set_online(DM._test_ind[0], DM._test_ind[-1], x_window_size)
    tst_batch_input = tst_batch["X"]
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w=torch.tensor(tst_batch_last_w,dtype=torch.float).cuda()
    tst_previous_w=torch.unsqueeze(tst_previous_w,1)  #(109, 11)-> [109,1,11]
    tst_batch_input=tst_batch_input.transpose((1,0,2,3))
    tst_batch_input=tst_batch_input.transpose((0,1,3,2))

    long_term_tst_src=torch.tensor(tst_batch_input,dtype=torch.float).cuda()
    tst_src_mask = (torch.ones(long_term_tst_src.size()[1],1,x_window_size)==1)  

    long_term_tst_currt_price=long_term_tst_src.permute((3,1,2,0))
    long_term_tst_currt_price=long_term_tst_currt_price[:,:,x_window_size-1:,:]

    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:,:,0:1,:],long_term_tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0,3,2,1))
    tst_trg_y = torch.tensor(tst_batch_y,dtype=torch.float).cuda()

    tst_long_term_w = []   #[2808-31,109,1,12]
    tst_y_window_size = len(DM._test_ind)-x_window_size-1-1
    for j in range(tst_y_window_size+1): #0-9
        tst_src = long_term_tst_src[:,:,j:j+x_window_size,:]
        tst_currt_price = long_term_tst_currt_price[:,:,j:j+1,:]
        if(local_context_length>1):
            padding_price = long_term_tst_src[:,:,j+x_window_size-1-local_context_length*2+2:j+x_window_size-1,:]
            padding_price = padding_price.permute((3,1,2,0))  
        else:
            padding_price = None
        out = model.forward(tst_src, tst_currt_price, tst_previous_w, tst_src_mask, tst_trg_mask, padding_price)
        if(j==0):
            tst_long_term_w = out.unsqueeze(0)  
        else:
            tst_long_term_w = torch.cat([tst_long_term_w,out.unsqueeze(0)],0)
        out = out[:,:,1:]  #去掉cash #[109,1,11]
        tst_previous_w = out
    tst_long_term_w = tst_long_term_w.permute(1,0,2,3) 
    tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO=evaluate_loss_compute(tst_long_term_w,tst_trg_y)  
    return tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO


def test_net(DM, total_step, output_step, x_window_size, local_context_length, model, loss_compute, evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ####每个epoch开始时previous_w=0
    max_tst_portfolio_value=0

    for i in range(total_step):        
        if(is_trn):
            loss, portfolio_value = train_one_step(DM, x_window_size, model, loss_compute, local_context_length)#如果包含优化器opt，会更新
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):   #每50个step输出一次
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                    (i,loss.item(), portfolio_value.item() , output_step / elapsed))
            start = time.time()
#########################################################tst########################################################   
        tst_total_loss=0
        torch.cuda.empty_cache()
#        print("torch.cuda.empty_cache()")
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                tst_loss, tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO = test_online(DM,x_window_size, model, evaluate_loss_compute, local_context_length)
                tst_total_loss += tst_loss.item()                                         
                elapsed = time.time() - start
                if(CR==0):
                    print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                            (i, tst_loss.item(), tst_portfolio_value.item() ,SR.item(), CR, TO.item(), 1/elapsed))
                else:
                    print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                            (i, tst_loss.item(), tst_portfolio_value.item() ,SR.item(), CR.item(), TO.item(), 1/elapsed))
                start = time.time()
#        portfolio_value_list.append(portfolio_value.item())
        
                if(tst_portfolio_value>max_tst_portfolio_value):
                    max_tst_portfolio_value=tst_portfolio_value
                    log_SR=SR
                    log_CR=CR
                    log_St_v=St_v
                    log_tst_pc_array=tst_pc_array
#                    model_dir="/data2/kaylakxu/PGPortfolio-master/PGPortfolio-master/model"
#                    torch.save(model_MLP, model_dir+'/'+'transformer_random_sample_pre_price_11_4_conv_multihead_asset_relation_only_encoder_pre_w.pkl')

#    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array
    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array,TO


def test_batch(DM,x_window_size,model,evaluate_loss_compute,local_context_length):
    tst_batch=DM.get_test_set()
    tst_batch_input = tst_batch["X"]
    tst_batch_y = tst_batch["y"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w,dtype=torch.float).cuda()
    tst_previous_w = torch.unsqueeze(tst_previous_w,1)
    tst_batch_input = tst_batch_input.transpose((1,0,2,3))
    tst_batch_input = tst_batch_input.transpose((0,1,3,2))

    tst_src = torch.tensor(tst_batch_input,dtype=torch.float).cuda()         
    tst_src_mask = (torch.ones(tst_src.size()[1],1,x_window_size)==1)   
    tst_currt_price = tst_src.permute((3,1,2,0))  
#############################################################################
    if(local_context_length>1):
        padding_price = tst_currt_price[:,:,-(local_context_length)*2+1:-1,:]  
    else:
        padding_price = None
#########################################################################
    tst_currt_price = tst_currt_price[:,:,-1:,:]  
    tst_trg_mask = make_std_mask(tst_currt_price,tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0,2,1))   #(109, 3, 11) ->#(109,11,3)
    tst_trg_y = torch.tensor(tst_batch_y,dtype=torch.float).cuda()
    tst_out = model.forward(tst_src, tst_currt_price, tst_previous_w, tst_src_mask, tst_trg_mask,padding_price)
    tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y) 
    return tst_loss, tst_portfolio_value


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index, loss_compute,evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    max_tst_portfolio_value=0
    for i in range(total_step):
        if(is_trn):
            loss, portfolio_value=train_one_step(DM,x_window_size,model,loss_compute,local_context_length)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):  
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                    (i,loss.item(), portfolio_value.item() , output_step / elapsed))
            start = time.time()
#########################################################tst########################################################     
        torch.cuda.empty_cache()
#        print("torch.cuda.empty_cache()")
        tst_total_loss=0
#        with torch.no_grad():
#            if(i % output_step == 0 and evaluate):
#                tst_loss, tst_portfolio_value=test_batch(DM,x_window_size,model,evaluate_loss_compute,filter_d)
##                tst_loss, tst_portfolio_value=evaluate_loss_compute(tst_out,tst_trg_y)
#                tst_total_loss += tst_loss.item()
#                elapsed = time.time() - start
#                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
#                        (i,tst_loss.item(), tst_portfolio_value.item() , 1/elapsed))
#                start = time.time()
                
#                if(tst_portfolio_value>max_tst_portfolio_value):
#                    max_tst_portfolio_value=tst_portfolio_value
    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
    print("save model!")



start = parse_time(FLAGS.start)
end = parse_time(FLAGS.end)
DM=DataMatrices(start=start,end=end,
             market="SP&500",
             feature_number=FLAGS.feature_number,      
             window_size=FLAGS.x_window_size,                            
             online=False,                            
             period=1800,                            
             coin_filter=FLAGS.coin_num,                            
             is_permed=False,                           
             buffer_bias_ratio=5e-5,                            
             batch_size=FLAGS.batch_size, #128,                          
             volume_average_days=30,                            
             test_portion=FLAGS.test_portion, #0.08,                  
             portion_reversed=False                            )





def make_model16(batch_size, coin_num, window_size, feature_number,N=6, 
               d_model_Encoder=512,d_model_Decoder=16, d_ff_Encoder=2048, d_ff_Decoder=64, h=8, dropout=0.1,local_context_length=3):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy   #深拷贝，拷贝对象及其子对象
    attn_Encoder = MultiHeadedAttention(True, h, d_model_Encoder,0.1,local_context_length)
    attn_Decoder = MultiHeadedAttention(False, h, d_model_Decoder,0.1,local_context_length)
    attn_En_Decoder = MultiHeadedAttention(False, h, d_model_Decoder,0.1,1)
    ff_Encoder = PositionwiseFeedForward(d_model_Encoder, d_ff_Encoder, dropout)
    ff_Decoder = PositionwiseFeedForward(d_model_Decoder, d_ff_Decoder, dropout)
    position_Encoder = PositionalEncoding(d_model_Encoder,0, dropout)
    position_Decoder = PositionalEncoding(d_model_Decoder, window_size-local_context_length*2+1,dropout)
    
    model = EncoderDecoder(batch_size, coin_num, window_size, feature_number,d_model_Encoder,d_model_Decoder,
        Encoder(EncoderLayer(d_model_Encoder, c(attn_Encoder), c(ff_Encoder), dropout), N),
        Decoder(DecoderLayer(d_model_Decoder, c(attn_Decoder), c(attn_En_Decoder), c(ff_Decoder), dropout), N),
        c(position_Encoder),                  #src position ecoding
        c(position_Decoder),                  #tgt position ecoding
        local_context_length              )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model             #返回一个EncoderDecoder类的实例




lr_model_sz=5120
factor=FLAGS.learning_rate  #1.0
warmup=0 #800

total_step=FLAGS.total_step
#num_epoch=100
x_window_size=FLAGS.x_window_size #31

batch_size=FLAGS.batch_size
coin_num=FLAGS.coin_num #11
feature_number=FLAGS.feature_number  #4
trading_consumption=FLAGS.trading_consumption #0.0025
variance_penalty=FLAGS.variance_penalty #0 #0.01
cost_penalty=FLAGS.cost_penalty #0 #0.01
output_step=FLAGS.output_step #50
local_context_length=FLAGS.local_context_length
model_dim=FLAGS.model_dim
weight_decay=FLAGS.weight_decay
interest_rate=FLAGS.daily_interest_rate/24/2

model_MLP = make_model16(batch_size, coin_num, x_window_size, feature_number,
                         N=1, d_model_Encoder=FLAGS.multihead_num*model_dim,
                         d_model_Decoder=FLAGS.multihead_num*model_dim, d_ff_Encoder=FLAGS.multihead_num*model_dim, 
                         d_ff_Decoder=FLAGS.multihead_num*model_dim, h=FLAGS.multihead_num, 
                         dropout=0.,local_context_length=local_context_length)

#model = make_model3(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
model_MLP = model_MLP.cuda()
#model_size, factor, warmup, optimizer)  用learn_rate表示warmup
model_opt = NoamOpt(lr_model_sz, factor, warmup, torch.optim.Adam(model_MLP.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9,weight_decay=weight_decay))

loss_compute=SimpleLossCompute2( Batch_Loss(trading_consumption,interest_rate,variance_penalty,cost_penalty,True), model_opt)
evaluate_loss_compute=SimpleLossCompute2( Batch_Loss(trading_consumption,interest_rate,variance_penalty,cost_penalty,False),  None)
test_loss_compute=SimpleLossCompute2_tst( Test_Loss(trading_consumption,interest_rate,variance_penalty,cost_penalty,False),  None)
#trn_loss, trn_PV=train_net(DM, total_step,output_step, window_size, model_MLP, loss_compute, evaluate_loss_compute, True, True)
#tst_portfolio_value, SR, CR, St_v,tst_pc_array=train_net(DM, total_step, output_step, x_window_size, model_MLP, loss_compute, evaluate_loss_compute, True, True)
train_net(DM, total_step, output_step, x_window_size, local_context_length ,model_MLP, FLAGS.model_dir, FLAGS.model_index, loss_compute, evaluate_loss_compute, True, True)

model_MLP=torch.load(FLAGS.model_dir+'/'+ str(FLAGS.model_index)+'.pkl')
tst_portfolio_value, SR, CR, St_v,tst_pc_array,TO=test_net(DM, 1, 1, x_window_size, local_context_length ,model_MLP, loss_compute, test_loss_compute, False, True)


if(CR!=0):
    CR=CR.item()

csv_dir=FLAGS.log_dir+"/"+"train_summary.csv"
d={"net_dir":[FLAGS.model_index],
    "fAPV":[tst_portfolio_value.item()],
    "SR":[SR.item()],
    "CR":[CR],
    "TO":[TO.item()],
    "St_v":[''.join(str(e)+', ' for e in St_v)],
    "backtest_test_history":[''.join(str(e)+', ' for e in tst_pc_array.cpu().numpy())],   
    }
new_data_frame = pd.DataFrame(data=d).set_index("net_dir")
if os.path.isfile(csv_dir):
    dataframe = pd.read_csv(csv_dir).set_index("net_dir")
    dataframe = dataframe.append(new_data_frame) 
else:
    dataframe = new_data_frame
dataframe.to_csv(csv_dir)