import _pickle as cp
import numpy as np

def load_dataset(filename):

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    x_train, y_train = data[0]
    x_val, y_val = data[1]
    x_test, y_test = data[2]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, val {1}, test {2}".format(x_train.shape, x_val.shape, x_test.shape))

    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return x_train, y_train, x_val, y_val, x_test, y_test


activities = ['Null', 'Open Door 1', 'Open Door 2' , 'Close Door 1' , 'Close door 2' , 'Open Fridge',
              'Close Fridge', 'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 
              'Close Drawer 1' ,'Open Drawer 2', 'Close Drawer 2', 'Open Drawer 3',
              'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']







#==============================================================================
# def evaluate (net, x_data, y_data, criterion, metric):
#         net.eval()
#         score = 0
#         all_pred = torch.LongTensor()
#         loss = 0
#         hidden = net.init_hidden(eval_batch_size)
#         for batch, i in enumerate(range(0, x_data.size(0) , seq_len)):
#             x_seq , y_seq = get_batch(x_data, y_data, i)
#             hidden = repackage_hidden( hidden)
#             output, hidden = net( x_seq, hidden )
#             output_flat = output.view(-1, num_classes)
#             y_seq = y_seq.view(-1)
#             loss += len(x_seq) * criterion( output_flat, y_seq).data
#             pred = output_flat.data.max(1)[1] #First 1 is dim (accross rows), second 1 is the argmax of each row
#             all_pred = torch.cat( (all_pred, pred.cpu() ))
#           #  print (y_seq)
#           #  print (all_pred)
#         global TEST
#         TEST = all_pred.numpy()
#         total_loss = loss[0]/len(x_data)
#         if metric == 'Acc':
#             score = (pred == y_data.data).cpu().sum() 
#         elif metric == 'F1':
#             score = f1_score( y_data.squeeze().cpu().numpy(), all_pred.cpu().numpy()  , average='macro')
#         return total_loss , score
# 
#==============================================================================