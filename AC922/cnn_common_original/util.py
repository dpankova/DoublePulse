# Common utility functions for training, using, and characterizing
# three-string double pulse ID CNNs
#
# Aaron Fienberg
# April 2020

import copy
import numpy as np
import tensorflow.keras as keras

# some default flux constants from
# https://docushare.icecube.wisc.edu/dsweb/Get/Document-87618/HESE_7.5_year_paper_v3.pdf
DEFAULT_INDEX = 2.88
DEFAULT_PHI = 2.1467


def normalized(data, in_place=False, method='per_image',
               mean=None, std=None):
    ''' normalize the images...
    can try different ways of doing this 
    method can be 'per_image' or 'whole_dataset'
    'per_image' masks out zeros for historical reasons

    'whole_dataset' requires mean and std input parameters, which must be 
    calculated external to this function
    '''

    if in_place:
        normalized_data = data
    else:
        normalized_data = np.empty_like(data, dtype=np.float32)

    if method == 'per_image':
        normalize_per_image(data, normalized_data)
    elif method == 'whole_dataset':
        normalized_data[:] = (data - mean)/std
    else:
        raise ValueError(f'method is \'{method}\', '
                         'but it must be \'per_image\' or '
                         ' \'whole_dataset\'!')

    return normalized_data


def normalize_per_image(data, output_array):
    ''' subtract mean, divide by RMS of nonzero samples '''
    for i, img in enumerate(data):
        nonzero = img[img != 0]
        if nonzero.size > 0:
            rms = np.std(nonzero)
            mean = np.mean(nonzero)
            output_array[i] = (img-mean) / rms
        else:
            output_array[i][:] = 0


def normalize_by_dataset(data, output_array, mean, std):
    ''' subtract mean, divide by std
        loops over imgs to keep memory usage from exploding
    '''
    for i, img in enumerate(data):
        output_array[i] = (img-mean) / rms


def load_dataset(e_file_list, tau_file_list,
                 n_training, n_validation,
                 img_shape,
                 get_weights=False,
                 weight_type='total',
                 spectral_index=DEFAULT_INDEX,
                 verbose=False):
    ''' loads all training and validation data from
    the provided file lists
    returns (training_data, validation_data), (e_f_list, tau_f_list)

    if get_weights is True, training data and validation will each be tuples
    of the form (images, weights), where the weights are relative weights 
    to account for the expected number of observed events in IceCube 

    weight_type: 'total' or 'energy'
    'total' uses the one-weight and the neutrino energy to determine the weight
    'energy' uses only the energy, ignoring the one weight

    spectral_index is only used if get_weights is true, in which case
    it determines the spectral index of the flux model used to generate
    the weights

    returned file lists contain the unconsumed file names
    '''

    output_arrays = []
    for n_to_load in [n_training, n_validation]:
        # load n_to_load for each flavor
        img_array = np.empty((2*n_to_load,) + img_shape,
                             dtype=np.float32)

        if get_weights:
            weights = np.empty(2*n_to_load, dtype=np.float32)

        out_e = load_n_images(e_file_list,
                              n_to_load,
                              img_array,
                              get_weights,
                              weight_type,
                              spectral_index,
                              verbose)
        e_file_list = out_e[-1]

        out_tau = load_n_images(tau_file_list,
                                n_to_load,
                                img_array[n_to_load:],
                                get_weights,
                                weight_type,
                                spectral_index,
                                verbose)
        tau_file_list = out_tau[-1]

        if get_weights:
            # copy weights into output array
            weights[:n_to_load] = out_e[1]
            weights[n_to_load:] = out_tau[1]

            output_arrays.append((img_array, weights))

        else:
            output_arrays.append(img_array)

    return tuple(output_arrays), (e_file_list, tau_file_list)


def load_n_images(file_list, n, img_array=None,
                  get_weights=False, weight_type='total',
                  spectral_index=DEFAULT_INDEX,
                  verbose=False):
    ''' returns (img_array, unread_file_list)
    where unread_file_list is file_list[last_read+1:]

    if img_array is not None, output will be copied 
    directly into img_array. Otherwise, a new array will be
    created to hold the output images.

    if get_weights is True, return value is instead
    (img_array, weights, unread_file_list), where 
    weights are the relative weights accounting for the expected number
    of observed events in IceCube assuming an astrophysical flux with 
    a spectral index of 'spectral_index'

    'weight_type': same as in load_dataset
    '''

    if img_array is not None and len(img_array) < n:
        raise RuntimeError('Provided output array is too small! '
                           f'Requested {n} images, but there is only enough '
                           f'room for {len(img_array)}.')

    if get_weights:
        weights = np.empty(n, dtype=np.float32)
        weights_view = weights

    n_copied = 0

    for i, file_name in enumerate(file_list):
        if verbose:
            print(f'reading {file_name}')

        with np.load(file_name) as np_file:
            batch = np_file['arr_0']
            if get_weights:
                if weight_type == 'total':
                    weights_batch = annual_weights_from_data(
                        batch, spectral_index)
                elif weight_type == 'energy':
                    weights_batch = energy_weights_from_data(
                        batch, spectral_index)
                else:
                    raise ValueError('weight_type must be \'total\''
                                     f' or \'energy\'. Got \'{weight_type}\'')

            batch = batch['image'][:, 0]

        if img_array is None:
            # determine image shape
            img_array = np.empty((n,) + batch[0].shape, dtype=np.float32)

        if i == 0:
            view = img_array[:n]

            if verbose:
                print(f'view shape: {view.shape}')

        if len(batch) <= len(view):
            view[:len(batch)] = batch
            view = view[len(batch):]

            if get_weights:
                weights_view[:len(batch)] = weights_batch
                weights_view = weights_view[len(batch):]

            n_copied += len(batch)

        else:
            view[:] = batch[:len(view)]

            if get_weights:
                weights_view[:] = weights_batch[:len(view)]

            n_copied += len(view)
            break

    if n_copied != n:
        raise RuntimeError(f'Requested {n} images, but found only {n_copied}!')

    if get_weights:
        return img_array, weights, file_list[i+1:]

    else:
        return img_array, file_list[i+1:]


def train_with_callback(model, initial_rate, momentum, patience,
                        n_epochs, callback,
                        dataset, batch_size,
                        input_adapter=None,
                        strategy=None,
                        optimizer='sgd',
                        opt_params=None):
    ''' input format:
        dataset: (training_data, validation_data)
        training_schedule: [(rate, momentum, n_epochs), ...]

        callback is a function that will be used to construct a 
        tf.keras.callbacks.LearningRateScheduler

        input_adapter: if this is not None, what is passed to 
        the model.fit is not img_array but input_adapter(img_array)

        returns history dictionary
    '''

    training_data, validation_data = dataset

    # check if we are using training weights
    if len(training_data) == 2:
        training_data, training_weights = training_data
        validation_data, validation_weights = validation_data
    else:
        training_weights = None

    n_train, n_valid = [int(len(data)/2)
                        for data in (training_data, validation_data)]

    training_labels, validation_labels = [
        np.array([0]*n + [1]*n) for n in [n_train, n_valid]]

    # prepare validation data/labels/weights
    if training_weights is not None:
        print('-----Using training weights!-----')
        validation_tuple = (validation_data,
                            validation_labels,
                            validation_weights)
    else:
        validation_tuple = (validation_data,
                            validation_labels)

    if strategy is not None:
        with strategy.scope():
            opt = build_optimizer(optimizer, opt_params,
                                  initial_rate, momentum)

            model.compile(
                loss='binary_crossentropy',
                optimizer=opt,
                metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])
    else:
        opt = build_optimizer(optimizer, opt_params,
                              initial_rate, momentum)

        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

    # build callbacks
    callbacks = []

    if callback is not None:
        callbacks.append(keras.callbacks.LearningRateScheduler(callback))

    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   restore_best_weights=True))
    callbacks.append(keras.callbacks.TensorBoard(profile_batch=0,
                                                 histogram_freq=3))
    # print LR callback from
    # https://www.tensorflow.org/tutorials/distribute/keras#keras_api

    class PrintLR(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(
                epoch + 1,
                model.optimizer.lr.numpy()))
    callbacks.append(PrintLR())

    print(f'Beginning learning rate scheduled training:')

    if input_adapter is not None:
        training_data = input_adapter(training_data)

        validation_data = input_adapter(validation_data)
        validation_tuple = (validation_data,) + validation_tuple[1:]

    return model.fit(
        x=training_data, y=training_labels,
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=callbacks,
        validation_data=validation_tuple,
        sample_weight=training_weights,
        shuffle=True).history


def build_optimizer(optimizer, opt_params, initial_rate=None, momentum=None):
    ''' returns an optimizer to be used in training
        initial_rate and momentum are redundant with opt_params,
        but for now I leave them to achieve backwards compatibility
    '''
    if optimizer == 'sgd':
        print('Using SGD')
        if initial_rate is not None or momentum is not None:
            return keras.optimizers.SGD(learning_rate=initial_rate,
                                        momentum=momentum)
        else:
            return keras.optimizers.SGD(**opt_params)
    elif optimizer == 'adam':
        print('Using Adam')
        return keras.optimizers.Adam(**opt_params)
    else:
        raise ValueError(f'\'{optimizer}\' is an unrecoginized optimizer!')


def get_all_scores(model, img_array, input_adapter=None):
    ''' classify all images in img_array
    returns a list of the obtained double_pulse scores

    input_adapter: same as for train_with_callback
    '''

    if input_adapter is not None:
        img_array = input_adapter(img_array)

    all_scores = model.predict(img_array)

    return all_scores[:, -1]


def make_summary_table(data, return_fields, output_table=None):
    ''' returns a summary table in which each column i contains
    something like data[return_fields[i]].
    for file_list in [e_files[-n_files[0]:], tau_files[-n_files[1]:]]:
        tables.append(np.vstack([test_model_on_file(model, file, 
                                                    input_adapter=input_adapter,
                                                    norm_mode=norm_mode, 
                                                    mean=mean, 
                                                    std=std,
                                                    verbose=True) 
                                 for file in file_list]))
    Iteratable return fields refer to nested keys/indices

    if output_table is not None, output data is placed directly into output_table
    otherwise, a new array is created
    '''
    if output_table is None:
        output_table = np.empty((len(data), len(return_fields)))

    for i, field_name in enumerate(return_fields):
        if type(field_name) is str:
            field_data = data[field_name]
        else:
            field_data = get_nested_field_data(data, field_name)

        output_table[:, i] = field_data.flatten()

    return output_table


def test_model_on_file(model,
                       file_name,
                       batch_size=256,
                       return_fields=[('weight', 'OneWeight'),
                                      ('weight', 'PrimaryNeutrinoEnergy')],
                       input_adapter=None,
                       norm_mode='per_image',
                       mean=None,
                       std=None,
                       verbose=False):
    '''classifies all events in the file using the provided model
       returns a summary table in which the first column contains
       the obtained scores and the second column contains the fields specified in 'return_fields'
       iterables in return_fields refer to nested keys/indices

       input_adapter: see train_with_callback
    '''

    if verbose:
        print(f'test_model_on_file: reading {file_name}')

    with np.load(file_name) as np_file:
        data = np_file['arr_0']

    # remove extra axis
    data = data.reshape(data.shape[0])

    n_rows = len(data)
    n_cols = 1 + len(return_fields)
    output_table = np.empty((n_rows, n_cols))

    # fill scores
    norm_img_array = normalized(data['image'], True,
                                norm_mode, mean, std)
    output_table[:, 0] = get_all_scores(model, norm_img_array, input_adapter)

    make_summary_table(data, return_fields, output_table[:, 1:])

    return output_table


def get_nested_field_data(data, field_name):
    for key in field_name:
        data = data[key]

    return data


def annual_weights_from_data(data, spectral_index=DEFAULT_INDEX):
    ''' get the annual weights from the original numpy array
        assuming the given spectral index
    '''
    table = make_summary_table(data,
                               [('weight', 'OneWeight'),
                                ('weight', 'PrimaryNeutrinoEnergy')])

    return get_annual_weights(table[:, 0], table[:, 1],
                              spectral_index=spectral_index)


def energy_weights_from_data(data, spectral_index=DEFAULT_INDEX):
    ''' returns weights for each event, where the weights are
        proportional to E**(-spectral_index). This function 
        ignores the one weights.

        To include one weights, use annual_weights_from_data
    '''
    table = make_summary_table(data,
                               [('weight', 'PowerLawIndex'),
                                ('weight', 'PrimaryNeutrinoEnergy')])

    # correct for index used in the simulation
    exponent = -(spectral_index - table[:, 0])

    return (table[:, 1]/100e3)**exponent


def get_annual_weights(one_weights,
                       nu_E,
                       n_files=1,
                       spectral_index=DEFAULT_INDEX,
                       phi_0=DEFAULT_PHI,
                       evts_per_file=10000):
    ''' returns the per-year weights for the given input parameters '''

    secs_per_year = 31536000

    total_events = 100*evts_per_file*n_files

    flux_weights = 1e-18*secs_per_year*phi_0*(nu_E/100e3)**(-spectral_index)

    return flux_weights/total_events*one_weights


def per_year_past_cut(table, score_cut, n_files):
    cut_table = table[table[:, 0] > score_cut]

    return get_annual_weights(cut_table[:, 1], cut_table[:, 2], n_files).sum()
