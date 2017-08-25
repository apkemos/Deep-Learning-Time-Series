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
#import pandas as pd
import time
import sys
#sns.set(color_codes=True)

num_classes = 18

batch_size = 128
eval_batch_size = 1
input_size = 113
seq_len = 180
hidden_size = 256
num_layers = 2

num_epochs = 100
lr = 0.001
lam_reg = 0
clip = 1

print_every = 50
append_every = 1
epoch_apend = 1
anneal_lr_every = 20
USE_CUDA = True
USE_CUDA_TEST = True
# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

model_path = os.getcwd() + '/net.pt'

epoch_break = 40
num_directions = 1




print("Loading data...")
x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np = data_utils.load_dataset('data/oppChallenge_gestures.data')

class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1*F.relu(-1*x)))
        return temp1 + temp2

class alpha_drop(nn.Module):
    def __init__(self, p = 0.05, alpha=-1.7580993408473766, fixedPointMean=0, fixedPointVar=1):
        super(alpha_drop, self).__init__()
        keep_prob = 1 - p
        self.a = np.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * pow(alpha-fixedPointMean,2) + fixedPointVar)))
        self.b = fixedPointMean - self.a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        self.alpha = alpha
        self.keep_prob = 1 - p
        self.drop_prob = p
    def forward(self, x):
        if self.keep_prob == 1 or not self.training:
            # print("testing mode, direct return")
            return x
        else:
            random_tensor  = self.keep_prob + torch.rand(x.size())
            
            binary_tensor = Variable(torch.floor(random_tensor))

            if torch.cuda.is_available():
                binary_tensor = binary_tensor.cuda()
            
            x = x.mul(binary_tensor)
            ret = x + self.alpha * (1-binary_tensor)
            ret.mul_(self.a).add_(self.b)
            return ret

class RNN(nn.Module):
    def __init__(self, lstm_input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.GRU(lstm_input_size, hidden_size, num_layers, batch_first=False, bidirectional=False, dropout = 0.5)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size*num_directions, num_classes)
        self.init_weights()
        self.sel = selu()
        self.a_drop = alpha_drop()
        
    
    def forward(self, x, hidden):
        # Set initial states 
        out, hidden = self.lstm(x, hidden)

    #    out = self.drop(out)

        decoder = self.fc( out.view( out.size(0) * out.size(1), out.size(2)) )
        decoder = decoder.view( out.size(0), out.size(1), num_classes )
        return decoder, hidden
    
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.fill_(0)
     #   self.fc.weight.data.uniform_(-initrange, initrange)
  #      self.fc.weight.data.normal_(0, initrange)
     #   weight_init.xavier_uniform(self.fc.weight.data, gain=nn.init.calculate_gain('tanh'))
        for name, param in self.named_parameters(): 
            if ('weight' in name):
                print ('Initializting ', name)
                weight_init.xavier_uniform(self.state_dict()[name], gain=nn.init.calculate_gain('tanh'))
        #        weight_init.orthogonal(self.state_dict()[name])
            #    self.state_dict()[name].normal_(0, initrange)
            #    print (self.state_dict()[name])
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers*num_directions, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*num_directions, batch_size, self.hidden_size).zero_()))
        
        


def batchify(data, batch_size):
    batch_len = data.shape[0] // batch_size
    refined_data = data[:batch_size * batch_len]
    refined_data = np.reshape(refined_data , (batch_size, batch_len, -1)).transpose((1,0,2))
    refined_data = np.ascontiguousarray(refined_data)
    return refined_data




x_train_np_batch = batchify(x_train_np, batch_size)
y_train_np_batch = batchify(y_train_np, batch_size)

batch_len = x_train_np_batch.shape[0]

x_train_np_batch = np.concatenate( (x_train_np_batch, x_train_np_batch) )
y_train_np_batch = np.concatenate( (y_train_np_batch, y_train_np_batch) )

x_train = torch.from_numpy(x_train_np_batch)
y_train = torch.from_numpy(y_train_np_batch).long()


x_val_np_batch = batchify(x_val_np, eval_batch_size)
y_val_np_batch = batchify(y_val_np, eval_batch_size)

x_val = torch.from_numpy( x_val_np_batch )
y_val = torch.from_numpy( y_val_np_batch ).long()
#x_val_var = Variable( x_val.view(x_val.size(0), 1, x_val.size(1)))
#y_val_var = Variable( y_val)


x_test = torch.from_numpy( batchify(x_test_np,1))
y_test = torch.from_numpy(y_test_np).long()


net = RNN(input_size, hidden_size, num_layers, num_classes)
if USE_CUDA == True:
    net.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=lam_reg)

if USE_CUDA == True:
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    
    x_val = x_val.cuda()
    y_val = y_val.cuda()
# =============================================================================
# x_test = x_test.cuda()
# y_test = y_test.cuda()    
# =============================================================================
def get_batch(source,labels, i, evaluation=False):
    bptt = min(seq_len, source.size(0) - i)
    data = Variable(source[i:i+bptt], volatile=evaluation)
    target = Variable(labels[i:i+bptt].squeeze())
    return data, target
    
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

num_train = x_train_np.shape[0]
iterations_per_epoch = x_train.size(1)//seq_len
#iterations = batches_per_epoch*num_epochs
samples_per_batch = x_train.size(1)

'''Todo: batchsize test'''



def evaluate (net, x_data, y_data, criterion, metric):
    net.eval()
    y_data = y_data.squeeze()
    x_data_var = Variable(x_data)
    y_data_var = Variable(y_data)
    if USE_CUDA_TEST == True:
        x_data_var = x_data_var.cuda()
        y_data_var = y_data_var.cuda()
    
    hidden, _ = net.init_hidden(x_data.size(1))
    output, hidden = net( x_data_var, hidden )
    
    loss = criterion( output.view(-1,num_classes), y_data_var)
    
    output_flat = output.view(-1,num_classes)
    output_flat = output_flat.data.max(1)[1] #First 1 is dim (accross rows), second 1 is the argmax of each row
    output_flat = output_flat[:,-1].cpu() #Reduce to 1d
 #   time.sleep(1)
    f1 = f1_score( y_data.cpu().numpy(), output_flat.numpy()  , average='macro')
    f1_w = f1_score( y_data.cpu().numpy(), output_flat.numpy()  , average='weighted')
    data_loss = loss.data[0] 
    return output_flat, data_loss, f1, f1_w

def evaluate_test (net, x_data, y_data, hidden, criterion, metric):
    net.eval()
    y_data = y_data.squeeze()
    x_data_var = Variable(x_data)
    y_data_var = Variable(y_data)
    if USE_CUDA_TEST == True:
        x_data_var = x_data_var.cuda()
        y_data_var = y_data_var.cuda()
    if hidden is None:
        hidden, _ = net.init_hidden(x_data.size(1))
    output, hidden = net( x_data_var, hidden )
    
    loss = criterion( output.view(-1,num_classes), y_data_var)
    
    output_flat = output.view(-1,num_classes)
    output_flat = output_flat.data.max(1)[1] #First 1 is dim (accross rows), second 1 is the argmax of each row
    output_flat = output_flat[:,-1].cpu() #Reduce to 1d
 #   time.sleep(1)
    f1 = f1_score( y_data.cpu().numpy(), output_flat.numpy()  , average='macro')
    f1_w = f1_score( y_data.cpu().numpy(), output_flat.numpy()  , average='weighted')
    data_loss = loss.data[0] 
    return output_flat, data_loss, f1, f1_w, hidden


time_per_epoch = 0
seqs_per_batch = batch_len // seq_len


    
best_val_score = None
better_model = True
epoch_counter = 0
epoch_of_best_model = 0
break_training = 0
break_flag = False
for epoch in range(1, num_epochs + 1):
    net.train()
    epoch_loss = 0
    start_time = time.time()  
    epoch_start = time.time()
    starting_index = np.random.randint(0, batch_len)
    hidden, _ = net.init_hidden(batch_size)
    if break_flag == True:
        break
    for batch, i in enumerate(range(starting_index, starting_index + batch_len , seq_len)):
        x_seq, y_seq = get_batch(x_train, y_train, i)
        Test1 = y_seq.cpu().data.numpy()
        y_seq = y_seq.view(-1)
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = net( x_seq, hidden )
        output_flat = output.view(-1,num_classes)
        loss = criterion(output_flat, y_seq )

        epoch_loss += loss.data[0]
        #loss = loss/bptt
        loss.backward()
        optimizer.step()
#        sys.exit()
        pred = output_flat.data.max(1)[1]
        score = f1_score( y_seq.data.cpu().numpy(), pred.cpu().numpy()  , average='macro')
        
#        torch.nn.utils.clip_grad_norm(net.parameters(), clip)
        if (batch % seqs_per_batch == 0) and (epoch % epoch_apend == 0):
            if (batch ==0 ): continue
            val_loss = 0              
         #   val_output = net( x_val_var )
            val_crit = nn.CrossEntropyLoss()
            _, val_loss , val_score, _ = evaluate(net, x_val, y_val, val_crit, 'F1')

     
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
                lr /= 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                epoch_counter = 0
            else: 
                break_training += 1
                epoch_counter += 1
                print ('Epoch increased {:d}'.format(epoch_counter) )
            
        if batch % print_every == 0:
            elapsed = time.time() - start_time
            print ('Epoch [%d/%d], Batch [%d/%d], Train Loss: %.4f, ms/batch %5.2f' 
                   %(epoch, num_epochs, batch, seqs_per_batch, loss.data[0], elapsed))
            start_time = time.time()
            
          
    epoch_loss /= seqs_per_batch    
    epoch_train_loss.append( epoch_loss )
    epoch_end = time.time()
    time_per_epoch += epoch_end - epoch_start
del x_train
del y_train
del x_val
del y_val

#net_loaded = net
net_loaded = None
if (better_model == True):
    net_loaded = RNN(input_size, hidden_size, num_layers, num_classes)
    net_loaded.load_state_dict(torch.load(model_path))
    if USE_CUDA_TEST == True:
        net_loaded.cuda()


print ('Evaluating on test data')
test_bs = 1


test_crit = nn.CrossEntropyLoss()
output_flat_test, test_loss, test_f1 , test_f1_w  = evaluate(net, x_test, y_test, test_crit, 'F1' )
all_preds = []
hidden_test = None
time.sleep(5)
for batch, i in enumerate(range(0, x_test.size(0), x_val_np.shape[0])):
    bsz = min(x_val_np.shape[0], x_test.size(0) - i)
    output_flat_test_, test_loss_, test_f1_ , test_f1_w_, hidden_test  = evaluate_test(net_loaded, x_test[i:i+bsz], y_test[i:i+bsz],hidden_test,
                                                                          test_crit, 'F1' )
    all_preds.append(output_flat_test_.cpu().numpy())
    
    
all_flat = np.concatenate(all_preds, axis=0)


y_test_preds = all_flat
y_test_preds_ = all_flat
test_f1_ = f1_score( y_test.cpu().numpy(), all_flat , average='macro')
test_f1_w_ = f1_score( y_test.cpu().numpy(), all_flat  , average='weighted')    


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
    orig_signals = np.zeros( (len(y_data), num_classes +1) )
    for i , label in enumerate (y_data):
        if (label != 0):
            orig_signals[i, label] = 1
     
    
    f, ax = plt.subplots(1)
    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=2)

    for i in range(num_classes):
        plt.plot(range(start_plot,end_plot), orig_signals[start_plot:end_plot,i], label= '[{:d}] - {}'.format(i, data_utils.activities[i]) )
    plt.legend(loc="upper left", bbox_to_anchor=[1, 1],
              ncol=1, shadow=True, title="Activities", fancybox=True)
    plt.show(f)        

start_plot = 3500
end_plot = 4200 
for i in range( x_test_np.shape[1]- 100):
    plt.plot(range(start_plot,end_plot), x_test_np[start_plot:end_plot,i] )    
    
plot_labels(y_test_np, start_plot, end_plot )
plot_labels(y_test_preds_, start_plot, end_plot )

start_plot = 0

#y_val_preds, _, _ , _ = evaluate(net_loaded, x_val, y_val, nn.CrossEntropyLoss(), 'F1')

#==============================================================================
# x_train_test = torch.from_numpy(x_train_np).view(-1, 1, 113).cuda() 
# y_train_test = torch.from_numpy(y_train_np).cuda().long()
# y_train_preds, train_loss,  train_f1, train_f1_w = evaluate(net, x_train_test, y_train_test, nn.CrossEntropyLoss(), 'F1')
# 
# plot_labels(y_train_np, start_plot, len(y_train_np))
# plot_labels(y_train_preds, start_plot, len(y_train_preds))
#==============================================================================

#plot_labels(y_val_preds, start_plot, len(y_val_preds))
#plot_labels(y_val_np, start_plot, len(y_val_np))

# =============================================================================
# plt.xlabel('Y_test_best_model')
# plt.ylabel('Label')
# =============================================================================
plot_labels(y_test_preds_, 0, len(y_test_preds) )
plot_labels(y_test_np, 0, len(y_test_np) )
#plot_labels(y_test_preds, 0, len(y_test_preds) )
#==============================================================================
# plot_labels(y_test_np, 0, 4000 )
# plot_labels(y_test_preds, 0, 4000)
#==============================================================================

start_plot = 58000
end_plot = 62000
channs_to_print = 5
for i in range(channs_to_print):
    plt.plot(range(start_plot,end_plot), x_test_np[start_plot:end_plot,i] )


plot_labels(y_test_preds_, start_plot, end_plot)
plot_labels(y_test_np, start_plot, end_plot )
#plot_labels(y_test_preds, start_plot, end_plot)
#==============================================================================
# plot_labels(y_test_np, 58000, 62000 )
# plot_labels(y_test_preds_, 58000, 62000)
#==============================================================================


# =============================================================================
# plt.figure(figsize = (10,7))
# cm = confusion_matrix(y_test_np, y_test_preds_)
# df_cm = pd.DataFrame(cm, index = [i for i in data_utils.activities],
#                   columns = [i for i in data_utils.activities])
#  
# 
# sns.heatmap(df_cm, annot=True,annot_kws={"size": 10}, fmt='d', robust=True)
# =============================================================================


print('Time per epoch {:.3f} seconds'.format(time_per_epoch/num_epochs) )
print('Train Last Loss {:.4f}'.format(train_losses[-1] ) )
print('Validation Last Loss {:.4f}'.format(val_losses[-1] ) )
print ('Best Train accuracy {0:1f} in epoch {1:1d}'.format(np.amax(train_acc_history), np.argmax(train_acc_history) + 1) )
print ('Best Validation accuracy {0:3f} in epoch {1:1d}'.format(np.amax(val_acc_history), np.argmax(val_acc_history) + 1 ) )
print('\nTest Loss {:.4f}'.format(test_loss_ ) )
print('F1 Score : {:.3f}'.format(test_f1) )
print('F1 Score best validation model : {:.3f}\n'.format(test_f1_) )

print('F1 Score weighted : {:.3f}'.format( test_f1_w_) )
print('F1 Score weighted best validation model : {:.3f}'.format( test_f1_w_) )