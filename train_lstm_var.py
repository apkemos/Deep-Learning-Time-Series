import torch 
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from numpy.lib.stride_tricks import as_strided as ast
import os
from torch.autograd import Variable
import data_utils
from sliding_window import sliding_window
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
from ggplot import *
import torch.nn.init as weight_init
from sklearn.utils import shuffle
#import pandas as pd
import time
import sys
import itertools
import graph
#sns.set(color_codes=True)

num_classes = 17

batch_size = 170 #221 130
eval_batch_size = 65
test_batch_size = 204
input_size = 113
seq_len = 180
hidden_size = 320
num_layers = 3
rnn_type = 'GRU'

num_epochs = 120
lr = 0.001
lam_reg = 0#1e-4
clip = 0.25

print_every = 5
append_every = 1
epoch_apend = 1
anneal_lr_every = 20
USE_CUDA = True
# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

model_path = os.getcwd() + '/net.pt'

epoch_break = 40





print("Loading data...")
x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np = data_utils.load_dataset('data/oppChallenge_gestures.data')







def batchify(data, batch_size):
    batch_len = data.shape[0] // batch_size
    refined_data = data[:batch_size * batch_len]
    refined_data = np.reshape(refined_data , (batch_size, batch_len, -1)).transpose((1,0,2))
    refined_data = np.ascontiguousarray(refined_data)
    return refined_data

def get_batch(source,labels, i, evaluation=False):
    bptt = min(seq_len, source.size(0) - i)
    data = Variable(source[i:i+bptt], volatile=evaluation)
    target = Variable(labels[i:i+bptt].squeeze())
    return data, target


def split_activities(data_x, data_y):
    activities_x = []
    activities_y = []
    i = 0
    train_len = data_y.shape[0]
    
    while (i< train_len):
        if data_y[i] == 0:
            i += 1
        else:
            act_x = [data_x[i]]
            act_y = [data_y[i]]
            i += 1
            while data_y[i] == data_y[i-1]  :
                act_x.append(data_x[i])
                act_y.append(data_y[i])

                i += 1
                if i == train_len: break
                
            activities_x.append(act_x)
            activities_y.append(act_y)
    
    return activities_x, activities_y



activities_x, activities_y = split_activities(x_train_np, y_train_np)
len_list_train = [len(act) for act in activities_y ]

activities_x, activities_y, len_list_train = shuffle(activities_x, activities_y, len_list_train)


activities_x_val, activities_y_val = split_activities(x_val_np, y_val_np)
len_list_val = [len(act) for act in activities_y_val ]

activities_x_test, activities_y_test = split_activities(x_test_np, y_test_np)
len_list_test = [len(act) for act in activities_y_test]




def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

test = None


def batchify_y(data, batch_size, len_list):
    if (len(data) % batch_size != 0):
        batch_num = len(data) // batch_size + 1
    else:
        batch_num = len(data) // batch_size
        
    batched_data = np.zeros((batch_num, max(len_list), batch_size), dtype=np.uint8)

    
    for i, batch_array in enumerate(chunks(data, batch_size)):
        batch_array = np.array(batch_array)

        if batch_array.shape[0] != batch_size:
           batch_array = np.lib.pad( batch_array , ( (0, batch_size- batch_array.shape[0]), (0,0) ), mode='constant', constant_values=(0,))
        batched_data[i] = batch_array.transpose()

    return batched_data



def batchify_x(data, batch_size, len_list):
    if (len(data) % batch_size != 0):
        batch_num = len(data) // batch_size + 1
    else:
        batch_num = len(data) // batch_size
        
    batched_data = np.zeros((batch_num, max(len_list), batch_size, NB_SENSOR_CHANNELS), dtype=np.float32)
    
    for i, batch_array in enumerate(chunks(data, batch_size)):

        batch_array = np.array( batch_array )
        print(batch_array.shape)
        if batch_array.shape[0] != batch_size:
           temp_array = np.zeros( (batch_size, max(len_list), NB_SENSOR_CHANNELS), dtype=np.float32 )

           print ('True')
           
           (dim_r, dim_c, _) = batch_array.shape
           temp_array[0:dim_r, 0:dim_c, :] = batch_array
           batched_data[i] = temp_array.transpose((1,0,2))
           break
       
       
        batched_data[i] = batch_array.transpose((1,0,2))
                                

    print (batched_data.shape)
    return batched_data



def batch_len_list(len_list, batch_size):
    if (len(len_list) % batch_size != 0):
        batch_num = len(len_list) // batch_size + 1
    else:
        batch_num = len(len_list) // batch_size
        
    batched_len_list = []
    gen = chunks(len_list, batch_size)
    for i in range(batch_num):
        batched_len_list.append( next(gen)  )
    return batched_len_list

batched_len_list_train = batch_len_list(len_list_train, batch_size)
batched_len_list_val = batch_len_list(len_list_val, eval_batch_size)
batched_len_list_test = batch_len_list(len_list_test, test_batch_size)

''' Append 0 to all activities until max_activitiy_len'''
def max_pad_activities(activities_x, activities_y, max_act_len):
    for t in activities_y:
        t.extend([np.uint8(0)] * (max_act_len - len(t)))        
            
    for t in activities_x:
        t.extend( [np.zeros(NB_SENSOR_CHANNELS, dtype=np.float32)] * (max_act_len - len(t)))
    return 

max_pad_activities(activities_x, activities_y, max(len_list_train))
max_pad_activities(activities_x_val, activities_y_val, max(len_list_val))
max_pad_activities(activities_x_test, activities_y_test, max(len_list_test))




sorted_lists = sorted(zip(len_list_train, activities_y, activities_x), reverse=True, key=lambda x: x[0])
len_list1, activities_y1, activities_x1 = [[x[i] for x in sorted_lists] for i in range(3)]



x_train_np_batch = batchify_x(activities_x, batch_size, len_list_train)
y_train_np_batch = batchify_y(activities_y, batch_size, len_list_train) - 1


x_train = torch.from_numpy(x_train_np_batch)
y_train = torch.from_numpy(y_train_np_batch).long()


del_inds = np.where( y_val_np == 0)[0]
y_val_np = np.delete( y_val_np, del_inds) - 1
x_val_np = np.delete( x_val_np, del_inds, axis=0)

x_val_np_batch = batchify_x(activities_x_val, eval_batch_size, len_list_val) 
y_val_np_batch = batchify_y(activities_y_val, eval_batch_size, len_list_val) - 1


x_val = torch.from_numpy( x_val_np_batch )
y_val = torch.from_numpy( y_val_np_batch ).long()


del_inds = np.where( y_test_np == 0)[0]
y_test_np = np.delete( y_test_np, del_inds) - 1 
x_test_np = np.delete( x_test_np, del_inds, axis=0)

x_test_np_batch = batchify_x(activities_x_test, test_batch_size, len_list_test)
y_test_np_batch = batchify_y(activities_y_test, test_batch_size, len_list_test) - 1


x_test = torch.from_numpy(x_test_np_batch)
y_test = torch.from_numpy(y_test_np_batch).long()

if USE_CUDA == True:
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    
    x_val = x_val.cuda()
    y_val = y_val.cuda()

    x_test = x_test.cuda()
    y_test = y_test.cuda()    




DEBUG = None
DEBUG_PAD = None
class RNN(nn.Module):
    def __init__(self, rnn_type, lstm_input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.recurrent_drop = 0.5
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(lstm_input_size, hidden_size, num_layers, batch_first=False, dropout=self.recurrent_drop)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes) 
        self.init_weights()
    #    self.sel = selu()
    #    self.a_drop = alpha_drop()
        
    
    def forward(self, x, hidden):
        # Set initial states 
        out, hidden = self.rnn(x, hidden)
        
        decoder_unp = None
        if ( isinstance(out, torch.nn.utils.rnn.PackedSequence) ):
            unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)
            out = out[0] #Make it out=out
            decoder_unp = self.fc( unpacked.view(-1, self.hidden_size))
            decoder_unp = decoder_unp.view( unpacked.size(0), unpacked.size(1), num_classes )
            

        global DEBUG
        global DEBUG_PAD
        DEBUG = out.data.cpu().numpy()


        DEBUG_PAD = out.data.cpu().numpy()
        decoder_packed = self.fc( out.view( -1 , self.hidden_size ) )



        return decoder_packed, decoder_unp, hidden
    
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.fill_(0)
     #   self.fc.weight.data.uniform_(-initrange, initrange)
  #      self.fc.weight.data.normal_(0, initrange)
     #   weight_init.xavier_uniform(self.fc.weight.data, gain=nn.init.calculate_gain('tanh'))
        for name, param in self.named_parameters(): 
            if ('weight' in name): #initiale with [- 1/sqrt(H) ,- 1/sqrt(H)]
                print ('Initializting ', name) 
          #      weight_init.xavier_uniform(self.state_dict()[name], gain=nn.init.calculate_gain('tanh'))
           #     weight_init.orthogonal(self.state_dict()[name])
            #    self.state_dict()[name].normal_(0, initrange)
                self.state_dict()[name].uniform_(-initrange, initrange)
             #   print (self.state_dict()[name])
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                        Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_() )
        
net = RNN(rnn_type, input_size, hidden_size, num_layers, num_classes)

if USE_CUDA == True:
    net.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=lam_reg)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

train_losses = []
val_losses = []
train_acc_history = []
val_acc_history = []
epoch_train_loss = []



def evaluate (net, x_data, y_data, batched_len_list, criterion,eval_batch_size, metric):
    net.eval()
    hidden = net.init_hidden(eval_batch_size)
    num_batches_activ = x_data.size(0) #x_data = Batch_num x largest activity x batch_size
    for i in range(num_batches_activ):
        x_batch = x_data[i]
        y_batch = y_data[i]
        
        len_list = batched_len_list[i]
        sorted_inds_vals = [t for t in sorted(enumerate(len_list), reverse=True, key=lambda x:x[1])]
        sorted_inds, sorted_vals = map(list, zip(*sorted_inds_vals))
        sorted_inds = np.array(sorted_inds)
        sorted_inds_t = torch.LongTensor(sorted_inds)
        if USE_CUDA == True:
            sorted_inds_t = sorted_inds_t.cuda()
            
        x_batch_s = torch.index_select(x_batch, 1, sorted_inds_t )
        y_batch_s = torch.index_select(y_batch, 1, sorted_inds_t )
        

        x_batch_s = Variable(x_batch_s)
        y_batch_s = Variable(y_batch_s)
        
        pack_x = torch.nn.utils.rnn.pack_padded_sequence(x_batch_s, sorted_vals) 
        pack_y = torch.nn.utils.rnn.pack_padded_sequence(y_batch_s, sorted_vals)

        y_true_non_zero = pack_y[0].view(-1) 
        output_packed, output, hidden = net( pack_x, hidden )

        loss = criterion(output_packed, y_true_non_zero )
        pred_flat = output_packed.data.max(1)[1]
        
        


    f1 = f1_score( y_true_non_zero.data.cpu().numpy(), pred_flat.cpu().numpy()  , average='macro')
    f1_w = f1_score( y_true_non_zero.data.cpu().numpy(), pred_flat.cpu().numpy()  ,  average='weighted')
    data_loss = loss.data[0] 
    return output_packed, output, data_loss, f1, f1_w, sorted_inds

time_per_epoch = 0


num_batches_activ = x_train_np_batch.shape[0]
    
best_val_score = None
better_model = False
epoch_counter = 0
epoch_of_best_model = 0
break_training = 0
break_flag = False
Test1 = None
for epoch in range(1, num_epochs + 1):
    net.train()
    epoch_loss = 0
    start_time = time.time()  
    epoch_start = time.time()
    hidden = net.init_hidden(batch_size)
    if break_flag == True:
        break
    for i in range(num_batches_activ):
        x_batch = x_train[i]
        y_batch = y_train[i]
        
        Y_BATCH = y_batch.cpu().numpy()

        len_list_batch = batched_len_list_train[i]
        sorted_inds_vals = [t for t in sorted(enumerate(len_list_batch), reverse=True, key=lambda x:x[1])]
        sorted_inds, sorted_vals = map(list, zip(*sorted_inds_vals))
        sorted_inds = np.array(sorted_inds)
        sorted_inds = torch.LongTensor(sorted_inds)
        if USE_CUDA == True:
            sorted_inds = sorted_inds.cuda()
   
        x_batch_s = torch.index_select(x_batch, 1, sorted_inds )
        y_batch_s = torch.index_select(y_batch, 1, sorted_inds )
        Y_BATCH_S = y_batch_s.cpu().numpy()


        x_batch_s = Variable(x_batch_s)
        y_batch_s = Variable(y_batch_s)
        pack_x = torch.nn.utils.rnn.pack_padded_sequence(x_batch_s, sorted_vals) 
        pack_y = torch.nn.utils.rnn.pack_padded_sequence(y_batch_s, sorted_vals)
        PACK_Y = pack_y[0].data.cpu().numpy()

        y_true_non_zero = pack_y[0].view(-1) 

        PACK_Y_NON_ZERO = pack_y[0].data.cpu().numpy()
        Y_TRUE_NON_ZERO = y_true_non_zero.data.cpu().numpy()

        optimizer.zero_grad()

        
        hidden = repackage_hidden(hidden)
        hidden = net.init_hidden(batch_size)
        output_packed, output, hidden = net( pack_x, hidden )
        OUTPUT = output.data.cpu().numpy()
        pack_out = torch.nn.utils.rnn.pack_padded_sequence(output, sorted_vals)
        loss = criterion(output_packed, y_true_non_zero )
        torch.nn.utils.clip_grad_norm(net.parameters(), clip)

        epoch_loss += loss.data[0]
        #loss = loss/bptt
        loss.backward()
        optimizer.step()

        
        pred_flat = output_packed.data.max(1)[1]
        pred_2d = output.data.max(2)[1]
        
        score = f1_score( y_true_non_zero.data.cpu().numpy(), pred_flat.cpu().numpy()  , average='macro')
        if (i==0):
            OUTPUT_I_0 = OUTPUT
            PRED_FLAT_I_0 = pred_flat.cpu().numpy()
            PACK_Y_I_0 = PACK_Y
            Y_TRUE_NON_ZERO_I_0 = Y_TRUE_NON_ZERO 
            Y_BATCH_I_0 = y_batch_s.data.cpu().numpy() 
            PREDS_FLAT = pred_flat.cpu().numpy()
            PREDS_2D = pred_2d.cpu().numpy()
            soft = nn.Softmax()
            PREDS_SOFTMAX = np.zeros( (output.cpu().data.numpy().shape))
            for i in range( output.size(0)):
                timestep_soft = soft( output[i])
                PREDS_SOFTMAX[i, : , :] = timestep_soft.data.cpu().numpy()
            TOTAL_SEQ_LOSS_OLD = loss.data[0]
            TOTAL_SEQ_LOSS = 0
            y_seq_label_concat = np.array(())
            decoder_unp_i0 = output
            ALL_ACT_LOSS = []
            for i, length in enumerate(sorted_vals):
                y_seq_pred  = output[0:length,i,:] #Get i activity sequence 
                y_seq_label = y_batch_s[0:length,i]
                seq_loss =  criterion(y_seq_pred, y_seq_label )
                TOTAL_SEQ_LOSS += seq_loss.data[0]
                y_seq_label_concat = np.concatenate((y_seq_label_concat, y_seq_label.data.cpu().numpy()))
                ALL_ACT_LOSS.append(seq_loss.data[0])
        if i % print_every == 0:
           
            elapsed = time.time() - start_time
            print ('Epoch [%d/%d], Batch [%d/%d], Train Loss: %.4f, ms/batch %5.2f' 
                   %(epoch, num_epochs, i, num_batches_activ, loss.data[0], elapsed))
            start_time = time.time()
        
#        torch.nn.utils.clip_grad_norm(net.parameters(), clip)
    if  (epoch % epoch_apend == 0):
        
        
        
        val_loss = 0              
        val_crit = nn.CrossEntropyLoss()
        pack_flat_val, pack_2d_val ,val_loss , val_score, _, s_ind = evaluate(net, x_val, y_val, batched_len_list_val, val_crit, eval_batch_size, 'F1')

        
        val_losses.append(val_loss)
        train_losses.append(loss.data[0])
        
        train_acc_history.append(score) #from last batch
        val_acc_history.append( val_score )  
        print ('Epoch [%d/%d] Validation Loss: %.4f' %(epoch, num_epochs, val_loss))
        if not best_val_score or val_score > best_val_score:
            best_val_score = val_score
            with open(model_path, 'wb') as f:
                torch.save(net.state_dict(), f)
            better_model = True
            epoch_counter = 0
            print ('New Best validation accuracy --------- {:.4f}-'.format(val_score))
            epoch_of_best_model = epoch
            break_training = 0
        elif break_training == epoch_break:
            break_flag = True #Make inner loop a train() function, return and break on outer loop
        elif epoch_counter == anneal_lr_every:
            print ('Learning rate reduced in epoch {:d}'.format(epoch))
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            epoch_counter = 0
        else: 
            break_training += 1
            epoch_counter += 1
            print ('Epoch increased {:d}'.format(epoch_counter) )
            
      
            
          
    epoch_loss /= num_batches_activ    
    epoch_train_loss.append( epoch_loss )
    epoch_end = time.time()
    time_per_epoch += epoch_end - epoch_start


net_loaded = net
if (better_model == True):
    net_loaded = RNN(rnn_type, input_size, hidden_size, num_layers, num_classes)
    net_loaded.load_state_dict(torch.load(model_path))
    if USE_CUDA == True:
        net_loaded.cuda()

print ('Evaluating on test data')
test_bs = 1


test_crit = nn.CrossEntropyLoss()
pack_flat_test, pack_2d_test, test_loss, test_f1 , test_f1_w, s_ind  = evaluate(net, x_test, y_test, batched_len_list_test,
                                                              test_crit, test_batch_size, 'F1' )
pack_flat_test_, pack_2d_test_, test_loss_, test_f1_ , test_f1_w_, s_ind_  = evaluate(net_loaded, x_test, y_test, batched_len_list_test,
                                                               test_crit,test_batch_size, 'F1' )




PACK_FLAT_TEST_ = pack_flat_test_.data.max(1)[1].cpu().numpy()

pred_test_2d_ = pack_2d_test_.data.max(2)[1]
pred_test_2d_ = np.squeeze(pred_test_2d_.cpu().numpy())
pred_test_unsorted_ = np.zeros( pred_test_2d_.shape, dtype=np.int64 )
pred_test_unsorted_[:, s_ind_] = pred_test_2d_
pred_test_unsorted_flat_ = [ pred_test_unsorted_[:length, col] for col ,length in enumerate(len_list_test)]
pred_test_unsorted_flat_stacked_ = np.hstack(pred_test_unsorted_flat_)


y_test_np_batch_sort = y_test_np_batch[:,:,s_ind_]
PREDS_2D_TEST = pack_2d_test_.data.cpu().numpy()
PREDS_2D_TEST_UNSORTED = PREDS_2D_TEST[:, s_ind_, :]
soft = nn.Softmax()
PREDS_SOFTMAX_TEST = np.zeros( (PREDS_2D_TEST.shape))
for i in range( PREDS_2D_TEST.shape[0]):
        timestep_soft = soft( Variable(torch.from_numpy(PREDS_2D_TEST[i])))
        PREDS_SOFTMAX_TEST[i, : , :] = timestep_soft.data.numpy()

y_test_preds = pred_test_unsorted_flat_stacked_
y_test_preds_ = pred_test_unsorted_flat_stacked_


y_compare = np.column_stack( (y_test_np, y_test_preds_) )

fig = plt.gcf()

plt.style.use('bmh')
plt.figure()
line_train, = plt.plot( train_acc_history , label='Train')
line_val, = plt.plot( val_acc_history , label='Validation')
plt.legend(handles=[line_train, line_val])
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.savefig(os.path.join(os.getcwd()+'/plots', 'Accuracy.png'))
plt.show()


plt.figure()
line_train, = plt.plot( train_losses, label='Train')
line_val, = plt.plot ( val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(handles=[line_train, line_val])
plt.savefig(os.path.join(os.getcwd()+'/plots', 'Loss.png'))
plt.show()



    
def plot_labels(y_data, start_plot, end_plot):
    orig_signals = np.zeros( (len(y_data), num_classes ) )
    for i , label in enumerate (y_data):
            orig_signals[i, label] = 1
     
    
    f, ax = plt.subplots(1)
    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=2)
    for i in range(0,num_classes):
        plt.plot(range(start_plot,end_plot), orig_signals[start_plot:end_plot,i], label= '[{:d}] - {}'.format(i, data_utils.activities[i+1]) )
    plt.legend(loc="upper left", bbox_to_anchor=[1, 1],
              ncol=1, shadow=True, title="Activities", fancybox=True)
    plt.show(f)        

start_plot = 0
end_plot = 1000 
signals_to_print = 3
for i in range(signals_to_print):
    plt.plot(range(start_plot,end_plot), x_test_np[start_plot:end_plot,i] )    
 
    
    
plot_labels(y_test_np, start_plot, end_plot )
plot_labels(y_test_preds_, start_plot, end_plot )


del_inds = np.where( y_train_np == 0)[0]
y_train_np_del = np.delete( y_train_np, del_inds) - 1
x_train_np_del = np.delete( x_train_np, del_inds, axis=0)
#plot_labels(y_train_np_del, start_plot, end_plot )

start_plot = 2800
end_plot = 3200
signals_to_print = 3
for i in range(signals_to_print):
    plt.plot(range(start_plot,end_plot), x_test_np[start_plot:end_plot,i] ) 
    
plot_labels(y_test_np, start_plot, end_plot )    
plot_labels(y_test_preds_, start_plot, end_plot ) 
#y_val_preds, _, _ , _ = evaluate(net_loaded, x_val, y_val, nn.CrossEntropyLoss(), 'F1')



start_plot = 0
end_plot = len(y_test_np)
# =============================================================================
# channs_to_print = 5
# for i in range(channs_to_print):
#     plt.plot(range(start_plot,end_plot), x_test_np[start_plot:end_plot,i] )
# =============================================================================
'''Plot each train batch label distribution'''
# =============================================================================
# for i in range( y_train.size(0) ):
#    plt.figure()
#    plt.hist(y_train_np_batch[i,0], bins=np.arange(y_train_np_batch[i,0].min(), y_train_np_batch[i,0].max()+2)-0.5, rwidth=0.5)
# =============================================================================

plot_labels(y_test_preds_, start_plot, end_plot) #put plot labels as argument
plot_labels(y_test_np, start_plot, end_plot )
plot_labels(y_test_preds, start_plot, end_plot)

plt.figure()
plt.hist(y_test_np, bins=np.arange(y_test_np.min(), y_test_np.max()+2)-0.5, rwidth=0.5)
plt.figure()
plt.hist(y_test_preds, bins=np.arange(y_test_preds_.min(), y_test_preds_.max()+3)-0.5, rwidth=0.5)
plt.figure()
plt.hist(y_test_preds_, bins=np.arange(y_test_preds.min(), y_test_preds.max()+3)-0.5, rwidth=0.5)

plt.figure()



plt.figure(figsize = (10,7))
cm = confusion_matrix(y_test_np, y_test_preds_)
df_cm = pd.DataFrame(cm, index = [i for i in data_utils.activities[1:]],
                  columns = [i for i in data_utils.activities[1:]])
 
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10}, fmt='d', robust=True)


print('Time per epoch {:.3f} seconds'.format(time_per_epoch/num_epochs) )
print('Train Last Loss {:.4f}'.format(train_losses[-1] ) )
print('Validation Last Loss {:.4f}'.format(val_losses[-1] ) )
print ('Best Train accuracy {0:1f} in epoch {1:1d}'.format(np.amax(train_acc_history), np.argmax(train_acc_history) + 1) )
print ('Best Validation accuracy {0:3f} in epoch {1:1d}'.format(np.amax(val_acc_history), np.argmax(val_acc_history) + 1 ) )
print('\nTest Loss {:.4f}'.format(test_loss ) )
print ('Best model accuracy {:.3f}'.format(np.sum( y_test_np == y_test_preds_)/ y_test_np.shape[0]))
print('F1 Score : {:.3f}'.format(test_f1) )
print('F1 Score best validation model : {:.3f}\n'.format(test_f1_) )

print('F1 Score weighted : {:.3f}'.format( test_f1_w) )
print('F1 Score weighted best validation model : {:.3f}'.format( test_f1_w_) )


# =============================================================================
# In the other python script , which you are going to import, you should put all the code that needs to be executed on running the script inside the following if block -
# 
# if '__main__' == __name__:
# =============================================================================
