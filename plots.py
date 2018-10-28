import numpy as np
import matplotlib.pyplot as plt

def plot_test(test_history):
  fig, axs = plt.subplots(2, 2, figsize=(21,10))

  fig.suptitle('Test History')

  epoch = np.arange(1,len(test_history['target_acc'])+1)
  axs[0][0].plot(epoch, np.array(test_history['target_acc']), label='Target prediction')
  axs[0][0].plot(epoch, np.array(test_history['source_acc']), label='Source prediction')
  axs[0][0].set_title('accuracy for class prediction')
  axs[0][0].set_ylabel('accuracy')
  axs[0][0].set_xlabel('epoch')
  axs[0][0].legend(frameon=True, facecolor='white')

  axs[0][1].plot(epoch, test_history['domain_acc'], label='Domain prediction')
  axs[0][1].set_title('accuracy for domain prediction')
  axs[0][1].set_ylabel('accuracy')
  axs[0][1].set_xlabel('epoch')
  axs[0][1].legend(frameon=True, facecolor='white')

  axs[1][0].plot(epoch, test_history['target_loss'], label='Target loss')
  axs[1][0].set_title('Average target domain test loss')
  axs[1][0].set_ylabel('loss')
  axs[1][0].set_xlabel('epoch')
  axs[1][0].legend(frameon=True, facecolor='white')

  axs[1][1].plot(epoch, test_history['source_loss'], label='Source loss')
  axs[1][1].set_title('Average source domain test loss')
  axs[1][1].set_ylabel('loss')
  axs[1][1].set_xlabel('epoch')
  axs[1][1].legend(frameon=True, facecolor='white')

def plot_training(train_history):
  fig, axs = plt.subplots(2, 2, figsize=(21,10))
  fig.suptitle('Train History')


  axs[0][0].plot(np.arange(len(train_history['train_loss'])), 
              train_history['train_loss'], label='Batch train loss')
  axs[0][0].set_title('Train loss over batches')
  axs[0][0].set_ylabel('loss')
  axs[0][0].set_xlabel('batch')
  axs[0][0].legend(frameon=True, facecolor='white')

  axs[0][1].plot(np.arange(len(train_history['domain_loss'])), 
              train_history['domain_loss'], label='Batch domain loss')
  axs[0][1].set_title('Domain loss over batches')
  axs[0][1].set_ylabel('loss')
  axs[0][1].set_xlabel('batch')
  axs[0][1].legend(frameon=True, facecolor='white')

  axs[1][0].plot(np.arange(len(train_history['avg_len_c'])), 
              train_history['avg_len_c'], label='Class Matrix Vector')
  axs[1][0].plot(np.arange(len(train_history['avg_len_d'])), 
              train_history['avg_len_d'], label='Domain Matrix Vector')
  axs[1][0].set_title('Avg len of vector')
  axs[1][0].set_ylabel('len of vector')
  axs[1][0].set_xlabel('batch')
  axs[1][0].legend(frameon=True, facecolor='white')

  axs[1][1].plot(np.arange(len(train_history['avg_dot'])), 
              train_history['avg_dot'], label='avg of matrices dot product')
  axs[1][1].set_title('Avg of dot product over batches')
  axs[1][1].set_ylabel('avg of dot')
  axs[1][1].set_xlabel('batch')
  axs[1][1].legend(frameon=True, facecolor='white')

def plot_domain_training(domain_train_history, train_history):
  fig, axs = plt.subplots(1, 3, figsize=(21,5))

  fig.suptitle('Train History')

  histLen = min(len(domain_train_history['avg_len']),
                len(train_history['avg_len_d']))
  model_gr_len_avg = train_history['avg_len_d'][:histLen]
  model_gr_dot_avg = train_history['avg_dot'][:histLen]

  axs[0].plot(np.arange(histLen), domain_train_history['avg_len'][:histLen],
                 label='Domain Net')
  axs[0].plot(np.arange(histLen), model_gr_len_avg,
                 label='Gradient Reverse Net')
  axs[0].set_title('Avg len of vector')
  axs[0].set_ylabel('len')
  axs[0].set_xlabel('batch')
  axs[0].legend(frameon=True, facecolor='white')

  axs[1].plot(np.arange(histLen), domain_train_history['avg_dot'][:histLen],
                 label='Domain Net')
  axs[1].plot(np.arange(histLen), model_gr_dot_avg,
                 label='Gradient Reverse Net')
  axs[1].set_title('Average of dot product with class net')
  axs[1].set_ylabel('dot product')
  axs[1].set_xlabel('batch')
  axs[1].legend(frameon=True, facecolor='white')

  axs[2].plot(np.arange(len(domain_train_history['avg_dot_gr'])),
              domain_train_history['avg_dot_gr'], label='Avg dot product')
  axs[2].set_title('Average of dot product with domain GR net')
  axs[2].set_ylabel('dot product')
  axs[2].set_xlabel('batch')
  axs[2].legend(frameon=True, facecolor='white')

def plot_multimodel_stats(trainers, test_histories):
  fig, axs = plt.subplots(2, 2, figsize=(21,10))
  fig.suptitle('Train History')
  for trainer_num, (single_trainer, test_history) in enumerate(list(zip(trainers, test_histories))): 
    train_history = single_trainer.train_history

    epoch = np.arange(1,len(test_history['target_acc'])+1)
    axs[0][0].plot(epoch, np.array(test_history['target_acc']), label='trainer {}'.format(trainer_num+1))
    axs[0][0].set_title('accuracy for class prediction')
    axs[0][0].set_ylabel('accuracy')
    axs[0][0].set_xlabel('epoch')
    axs[0][0].legend(frameon=True, facecolor='white')
     
    axs[0][1].plot(np.arange(len(train_history['train_loss'])), 
                train_history['train_loss'], label='trainer {}'.format(trainer_num+1))
    axs[0][1].set_title('Train loss over batches')
    axs[0][1].set_ylabel('loss')
    axs[0][1].set_xlabel('batch')
    axs[0][1].legend(frameon=True, facecolor='white')

    vec_len_sum = np.array(train_history['avg_len_c']) + np.array(train_history['avg_len_d'])
    axs[1][0].plot(np.arange(len(train_history['avg_len_c'])), vec_len_sum,
                   label='trainer {}'.format(trainer_num+1))
    axs[1][0].set_title('Avg len of vectors')
    axs[1][0].set_ylabel('len of vector')
    axs[1][0].set_xlabel('batch')
    axs[1][0].legend(frameon=True, facecolor='white')

    axs[1][1].plot(np.arange(len(train_history['avg_dot'])), 
                train_history['avg_dot'], label='trainer {}'.format(trainer_num+1))
    axs[1][1].set_title('Avg of dot product over batches')
    axs[1][1].set_ylabel('avg of dot')
    axs[1][1].set_xlabel('batch')
    axs[1][1].legend(frameon=True, facecolor='white')
