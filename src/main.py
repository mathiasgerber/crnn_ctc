import os
import argparse

import editdistance

from dataloader import DataLoader
from model import ModelValues, Model

my_path = os.path.abspath(os.path.dirname(__file__))


class FilePaths:
    """
    Paths and names to the data. (os.path.join is used for Linux-Compatibility)
    """
    fn_char_list = os.path.join(my_path, '../../data/char_list')
    fn_train = os.path.join(my_path, '../../data/')
    fn_corpus = os.path.join(my_path, '../../data/corpus.txt')
    fn_accuracy_md = os.path.join(my_path, '../model/accuracy_md.txt')
    fn_accuracy_od = os.path.join(my_path, '../model/accuracy_od.txt')


def main():
    """
    Main function that takes in Arguments to run the appropriate functions,
    train or validation on test set
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_md', help='Train 2D-LSTM',
                        action='store_true')
    parser.add_argument('--train_od', help='Train 1D-LSTM',
                        action='store_true')
    parser.add_argument('--validate',
                        help='validate the NN with our own data',
                        action='store_true')

    args = parser.parse_args()

    loader = DataLoader(FilePaths.fn_train, ModelValues.batch_size,
                        ModelValues.image_size, ModelValues.max_text_length)
    open(FilePaths.fn_char_list, 'w').write(str().join(loader.char_list))

    open(FilePaths().fn_corpus, 'w').write(str(' ').join(loader.train_words))

    if args.train_md:
        model = Model(loader.char_list, True)
        train(model, loader, True)
    elif args.train_od:
        model = Model(loader.char_list, False)
        train(model, loader, False)
    else:
        print("Wrong argument.")


def train(model, loader, md_bool):
    """
    Function that trains Batches in a while loop.
    :param model: The Model which will be trained
    :param loader: The Dataloader which gets the Batchdata.
    :return: nothing. (Stops if there was no progress for 5 epochs.)
    """
    epoch = 0
    best_char_error_rate = float('inf')
    no_improvement_since = 0
    early_stopping = 20

    while True:
        epoch += 1
        print('Epoch:', epoch)

        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            loss = model.train_batch(batch)
            print('Batch:', iter_info[0], '/', iter_info[1], 'Loss:', loss)

        char_error_rate = validate(model, loader)

        if char_error_rate < best_char_error_rate:
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save(md_bool)
            if md_bool:
                open(FilePaths.fn_accuracy_md, 'w').write('Validation '
                                                          'character'
                                                          ' error rate of '
                                                          'saved md_model: '
                                                          '%f%%' %
                                                          (char_error_rate*
                                                           100.0))
            else:
                open(FilePaths.fn_accuracy_od, 'w').write('Validation '
                                                          'character'
                                                          ' error rate of '
                                                          'saved od_model: '
                                                          '%f%%' %
                                                          (char_error_rate *
                                                           100.0))

        else:
            no_improvement_since += 1

        if no_improvement_since >= early_stopping:
            print('Training stopped.')
            break


def validate(model, loader):
    """
    Function to validate the Model on the validation set. Before returning,
    the character error rate and the word accuracy get printed.
    :param model: The Model which will be validated
    :param loader: The Dataloader
    :return: Returns the character error rate.
    """
    loader.validation_set()
    num_char_error = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print('Batch:', iter_info[0], '/', iter_info[1])
        batch = loader.get_next()
        (recognized, _) = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_error += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.gt_texts[i] + '"', '->', '"' + recognized[i] + '"')

    char_error_rate = num_char_error / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print('Character error rate: %f%%. Word accuracy: %f%%.' %
          (char_error_rate*100.0, word_accuracy*100.0))
    return char_error_rate


if __name__ == '__main__':
    main()
