# Contains driver function for training and characterizing a new model
#
# Aaron Fienberg
# April 2020

from cnn_common.plotting import *
from cnn_common.util import *
import time
import sys
import tensorflow as tf
import tensorflow.keras as keras

from glob import glob

sys.path.append('..')


def build_and_train_model(n_training,
                          n_validation,
                          initial_rate,
                          momentum,
                          n_epochs,
                          patience,
                          batch_size,
                          scheduler,
                          validation_files_per_flavor,
                          e_file_list,
                          tau_file_list,
                          model_factory,
                          input_adapter=None,
                          use_weights=False,
                          norm_mode='per_image',
                          weight_type='total',
                          spectral_index=DEFAULT_INDEX,
                          optimizer=None,
                          opt_params=None):
    '''
    builds and trains a model according to the given parameters

    returns the trained model
    '''

    strategy = tf.distribute.MirroredStrategy()

    print('\nbuilding the model...\n')

    with strategy.scope():
        model = model_factory()

    model.summary()

    print('\nLoading raw data...\n')
    start = time.time()
    dataset, unconsumed_files = load_dataset(e_file_list, tau_file_list,
                                             n_training, n_validation,
                                             (500, 60, 3),
                                             get_weights=use_weights,
                                             spectral_index=spectral_index,
                                             weight_type=weight_type,
                                             verbose=True)

    if use_weights:
        # if we are using training weights, rescale them
        # so the average value is one
        for weights in [img_set[1] for img_set in dataset]:
            weights /= np.mean(weights)
        print(f'average training weight: {np.mean(dataset[0][1])}')
        print(f'training weight sum: {np.sum(dataset[0][1])}')

        print(f'average validation weight: {np.mean(dataset[1][1])}')
        print(f'validation weight sum: {np.sum(dataset[1][1])}')

    elapsed = time.time() - start
    print(f'loading raw data took {elapsed:.1f} seconds')

    print('\nNormalizing...\n')
    start = time.time()

    mean, std = None, None
    if norm_mode == 'whole_dataset':
        if use_weights:
            mean = np.mean(dataset[0][0])
            std = np.std(dataset[0][0])
        else:
            mean = np.mean(dataset[0])
            std = np.std(dataset[0])
        with open('dataset_norm_stats.txt', 'w') as stats_f:
            stats_f.write(f'mean: {mean}\n')
            stats_f.write(f'std: {std}\n')

    if use_weights:
        # if using weights, dataset will be a list of tuples
        # of the form (img_array, img_weights)
        # we must retain that format
        dataset = [(normalized(img_set[0], True, norm_mode,
                               mean, std), img_set[1])
                   for img_set in dataset]
    else:
        # if we're not using weights, dataset will just be a list
        # of img arrays
        dataset = [normalized(array, True, norm_mode,
                              mean, std) for array in dataset]

    elapsed = time.time() - start
    print(f'normalization took {elapsed:.1f} seconds')

    print('\nTraining...\n')
    start = time.time()
    combined_history = train_with_callback(
        model,
        initial_rate, momentum,
        patience, n_epochs,
        scheduler,
        dataset,
        batch_size,
        input_adapter,
        strategy=strategy,
        optimizer=optimizer,
        opt_params=opt_params)

    elapsed = time.time() - start
    print(f'training took {elapsed:.1f} seconds')

    plot_history(combined_history)

    #print('\nCharacterizing performance...\n')

    #characterization_files = [
    #    file_list[:n_files]
    #    for file_list, n_files in zip(unconsumed_files,
    #                                  validation_files_per_flavor)]

    #characterize_model(model, *characterization_files, input_adapter,
    #                   norm_mode, mean, std)

    return model


def characterize_model(model, e_file_list, tau_file_list, input_adapter=None,
                       norm_mode='per_image', mean=None, std=None):
    ''' characterizes an already trained model 
        on the provided nu_e and nu_tau files'''
    tables = []

    print('building summary tables...')
    start = time.time()

    for file_list in [e_file_list, tau_file_list]:
        tables.append(
            np.vstack([test_model_on_file(model, file,
                                          input_adapter=input_adapter,
                                          norm_mode=norm_mode,
                                          mean=mean,
                                          std=std,
                                          verbose=True)
                       for file in file_list]))

    elapsed = time.time() - start
    print(f'building tables took {elapsed:.1f} seconds')

    validation_files_per_flavor = [len(l)
                                   for l in [e_file_list, tau_file_list]]

    n_per_year_plot(tables, validation_files_per_flavor)

    for cut in [0, 0.9, 0.95, 0.99]:
        print('-----')
        print(f'cut: {cut}')
        for table, n_files, flav in zip(tables,
                                        validation_files_per_flavor,
                                        ['e', 'tau']):
            n_per_year = per_year_past_cut(table, cut, n_files)
            print(f'n nu_{flav} per year: {n_per_year:.3f}')
        print('-----')
