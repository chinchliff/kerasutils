import numpy as np
import random, os, time

def train_test_split(X, y, test_size=0.3):
    '''Group test seqs into batches by length and create test/train sets.'''
    '''Assumes incoming training data are randomly ordered--does no randomization.'''
    
    t = range(len(X))
    
    ntrn = round(len(X) * (1 - test_size))
    X_train, y_train, X_test, y_test = ({}, {}, {}, {})
    for i in t:
        k = len(X[t[i]])
        if i < ntrn:
            if k not in X_train:
                X_train[k] = []
                y_train[k] = []
            X_train[k].append(X[t[i]])
            y_train[k].append(y[t[i]])
        else:
            if k not in X_test:
                X_test[k] = []
                y_test[k] = []
            X_test[k].append(X[t[i]])
            y_test[k].append(y[t[i]])
    
    return (X_train, y_train), (X_test, y_test)

def write_model_description(model, dataset, filename):
    with open(filename, 'w') as outfile:
        outfile.write('dataset: {f}\n\n'.format(f=dataset))

        m = model.get_config()
        outfile.write('loss: {l}\n'.format(l=m['loss']))
        outfile.write('class_mode: {m}\n'.format(m=m['class_mode']))
        opt_string = ', '.join(['{k}: {v}'.format(k=str(k), v=str(v)) for k, v in m['optimizer'].items()])
        outfile.write('optimizer: [{o}]\n'.format(o=opt_string))
        
        for i, l in enumerate(m['layers']):
            outfile.write('{i}: {name}\n'.format(i=i, name=l['name']))
            for k in sorted(l.keys()):
                outfile.write('    {k} = {v}\n'.format(k=k, v=str(l[k])))
        outfile.flush()

def train_classification_model(model, X, y, run_label, dataset, num_epochs, \
                               train_scoring_proportion, resume_previous=None):

    start_epoch = 0
    if resume_previous is None:
        output_dir = 'results_{l}_{t}/'.format(l=run_label, t=int(time.time()))
        os.mkdir(output_dir)
    else:
        if not os.path.exists(resume_previous):
            raise OSError('Cannot resume previous run \'{}\' because it cannot be found' \
                          .format(resume_previous))
        output_dir = resume_previous.rstrip('/') + '/'
        # use the weight file with the highest epoch number as the starting point
        
#        print(sorted([int(t[5:-8]) for t in os.listdir(output_dir) if t[-7:] == 'weights']))
        
#        last_epoch_completed = sorted([int(t[5:-8]) for t in os.listdir(output_dir) if t[-7:] == 'weights'])[-1]

        print(last_epoch_completed)
        model.load_weights(output_dir + 'epoch' + str(last_epoch_completed) + '.weights')
        if num_epochs <= last_epoch_completed:
            raise ValueError(('The number of epochs ({}) must be greater than the' +
                              'number that have already been completed ({}).') \
                             .format(num_epochs, last_epoch_completed))
        start_epoch = last_epoch_completed + 1

#    print(start_epoch)
    write_model_description(model, dataset, output_dir + '_model.description.txt')

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y)
    num_outputs = y_test.values()[0][0].shape[0]

    score_file_path = output_dir + '_scores.csv'
    if resume_previous is None:
        with open(score_file_path, 'w') as score_file:
            score_file.write('epoch,train_err_abs,test_err_abs,train_err_mean,test_err_mean\n')

    for cur in range(start_epoch, num_epochs):
        print('epoch {}'.format(cur))

        i = 1
        for k, batch in X_train.items():
            print('training on {n} seqs of length {k}. batch {i}, {m} remaining'.format( \
                    n=len(batch), k=k, i=i, m=len(X_train)-i))
            model.train_on_batch(np.array(np.array(batch)), np.array(y_train[k]))
            i += 1

        # predict and score the test data
        error_test = 0
        n_test_seqs = 0
        with open(output_dir + 'test_result_epoch{cur}.csv'.format(cur=cur), 'w') as outfile:
            par_labels = ['p' + str(i) for i in range(num_outputs)]
            col_labels = [x for t in zip([p + '_simulated' for p in par_labels], \
                                         [p + '_predicted' for p in par_labels]) for x in t]
            outfile.write(','.join(col_labels) + ',loss,size\n')

            for i, k in enumerate(X_test.keys()):
                batch_X = X_test[k]
                batch_y = y_test[k]
                n_k = len(batch_X)
                p = model.predict(np.array(batch_X))
                n_test_seqs += n_k
                print('predicting inputs for {n} test seqs of length {k}. batch {i}, {m} remaining' \
                    .format(n=n_k, k=k, i=i+1, m=len(X_test) - i - 1))

                for l in range(n_k):
                    X_l = np.array([batch_X[l],])
                    y_l = np.array([batch_y[l],])
                    n_classes = len(y_l[0])
                    d = model.evaluate(X_l, y_l)
                    error_test += d

                    # predict class, convert to array representation # this method is not working
#                    p = model.predict_classes(X_l)
#                    p_arr = [0,] * len(batch_y[l])
#                    p_arr[p] = 1
#                    p = p_arr

                    # predict continuous scores for classes, convert to array representation
                    p_arr = [0,] * n_classes
                    if n_classes > 1:
                        p_arr[np.where(p[l] == p[l].max())[0][0]] = 1
                        p_l = p_arr
                    else: # binary classification
                        p_l = np.around(p[l])

                    # output
                    par_vals = [x for t in zip([str(y) for y in y_l[0]], \
                                               [str(y) for y in p_l]) for x in t]
                    outfile.write(','.join(par_vals) + ',' + \
                                  str(d) + ',' + \
                                  str(len(batch_X[l])) + '\n')


        # also predict and score (some of the) training data
        error_train = 0
        n_train_seqs = 0
        X_train_subsample = random.sample(X_train.keys(), int(len(X_train) * train_scoring_proportion) + 1)
        for i, k in enumerate(X_train_subsample):
            batch_X = X_train[k]
            batch_y = y_train[k]
            n_k = len(batch_X)
            n_train_seqs += n_k
            print(('predicting inputs for {n} training seqs of length {k}. ' +
                  'batch {i}, {m} remaining') \
                  .format(n=n_k, k=k, i=i+1, m=len(X_test) - i - 1))

            for l in range(n_k):
                X_l = np.array([batch_X[l],])
                y_l = np.array([batch_y[l],])
                error_train += model.evaluate(X_l, y_l)

        # calculate mean distances
        error_train_mean = error_train / float(n_train_seqs)
        error_test_mean = error_test / float(n_test_seqs)
        
        with open(score_file_path, 'a') as score_file:
            score_file.write(','.join([str(x) for x in \
                [cur, error_train, error_test, error_train_mean, error_test_mean]]) + '\n')

        print('epoch {cur} train error | abs: {ea:.1f} | mean: {es:.4f}' \
                .format(cur=cur, ea=error_train, es=error_train_mean))
        print('epoch {cur} test  error | abs: {ea:.1f} | mean: {es:.4f}' \
                .format(cur=cur, ea=error_test, es=error_test_mean))

        # record the weights found this epoch
        model_file = output_dir + 'epoch{cur}.weights'.format(cur=cur)
        model.save_weights(model_file)

def train_regression_model(model, X, y, run_label, dataset, num_epochs, \
                           train_scoring_proportion, resume_previous=None):

    start_epoch = 0
    if resume_previous is None:
        output_dir = 'results_{l}_{t}/'.format(l=run_label, t=int(time.time()))
        os.mkdir(output_dir)
    else:
        if not os.path.exists(resume_previous):
            raise OSError('Cannot resume previous run \'{}\' because it cannot be found' \
                          .format(resume_previous))
        output_dir = resume_previous.rstrip('/') + '/'
        # use the weight file with the highest epoch number as the starting point
        
        print(sorted([t[5:-8] for t in os.listdir(output_dir) if t[-7:] == 'weights']))
        
        last_epoch_completed = sorted([int(t[5:-8]) for t in os.listdir(output_dir) if t[-7:] == 'weights'])[-1]
        model.load_weights(output_dir + 'epoch' + str(last_epoch_completed) + '.weights')
        if num_epochs <= last_epoch_completed:
            raise ValueError(('The number of epochs ({}) must be greater than the' +
                              'number that have already been completed ({}).') \
                             .format(num_epochs, last_epoch_completed))
        start_epoch = last_epoch_completed + 1

    write_model_description(model, dataset, output_dir + '_model.description.txt')

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y)
    num_outputs = y_test.values()[0][0].shape[0]

    score_file_path = output_dir + '_scores.csv'
    if resume_previous is None:
        with open(score_file_path, 'w') as score_file:
            score_file.write('epoch,train_err_abs,test_err_abs,train_err_mean,test_err_mean\n')

    for cur in range(num_epochs):
        print('epoch %d' % cur)

        i = 1
        for k, batch in X_train.items():
            print('training on {n} seqs of length {k}. batch {i}, {m} remaining' \
                    .format(n=len(batch), k=k, i=i, m=len(X_train)-i))
            model.train_on_batch(np.array(np.array(batch)), np.array(y_train[k]))
            i += 1

        # predict and score the test data
        error_test = 0
        n_test_seqs = 0
        with open(output_dir + 'test_result_epoch{cur}.csv'.format(cur=cur), 'w') as outfile:
            par_labels = ['p' + str(i) for i in range(num_outputs)]
            col_labels = [x for t in zip([p + '_simulated' for p in par_labels], \
                                         [p + '_predicted' for p in par_labels]) for x in t]
            outfile.write(','.join(col_labels) + ',loss,size\n')

            for k in y_test:
                batch_X = X_test[k]
                batch_y = y_test[k]
                p = model.predict(np.array(batch_X))
                n_k = len(batch_X)
                n_test_seqs += n_k
                print('predicting inputs for {n} test seqs of length {k}. batch {i}, {m} remaining' \
                    .format(n=len(batch), k=k, i=i+1, m=len(X_test) - i - 1))
                
                for l in range(n_k):
                    X_l = np.array([batch_X[l],])
                    y_l = np.array([batch_y[l],])
                    d = model.evaluate(X_l, y_l)
                    error_test += d
                    par_vals = [x for t in zip([str(y) for y in y_l[0]], \
                                               [str(y) for y in p[l]]) for x in t]
                    outfile.write(','.join(par_vals) + ',' + \
                                  str(d) + ',' + \
                                  str(len(batch_X[l])) + '\n')

        # also predict and score (some of the) training data
        error_train = 0
        n_train_seqs = 0
        X_train_subsample = random.sample(X_train.keys(), int(len(X_train) * train_scoring_proportion))
        for i, k in enumerate(X_train_subsample):
            batch_X = X_train[k]
            batch_y = y_train[k]
            n_k = len(batch_X)
            n_train_seqs += n_k
            print('predicting inputs for {n} training seqs of length {k}. batch {i}, {m} remaining' \
                    .format(n=n_k, k=k, i=i+1, m=len(X_train_subsample) - i - 1))

            for l in range(n_k):
                X_l = np.array([batch_X[l],])
                y_l = np.array([batch_y[l],])
                error_train += model.evaluate(X_l, y_l)

        # calculate mean distances
        error_train_mean = error_train / float(n_train_seqs)
        error_test_mean = error_test / float(n_test_seqs)

        with open(score_file_path, 'a') as score_file:
            score_file.write(','.join([str(x) for x in \
                [cur, error_train, error_test, error_train_mean, error_test_mean]]) + '\n')

        print('epoch {cur} train error | abs: {ea:.9f} | mean: {es:.9f}' \
                .format(cur=cur, ea=error_train, es=error_train_mean))
        print('epoch {cur} test  error | abs: {ea:.9f} | mean: {es:.9f}' \
                .format(cur=cur, ea=error_test, es=error_test_mean))

        # record the weights found this epoch
        model_file = output_dir + 'epoch{cur}.weights'.format(cur=cur)
        model.save_weights(model_file)
