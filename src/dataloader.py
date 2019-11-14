import os
import random

import numpy as np
import cv2

from preprocessor import preprocess


class Sample:
    """
    Sample from the dataset.
    """
    def __init__(self, gt_text, file_path):
        self.gt_text = gt_text
        self.file_path = file_path


class Batch:
    """
    Batch containing images and ground truth texts.
    """
    def __init__(self, gt_texts, images):
        self.images = np.stack(images, axis=0)
        self.gt_texts = gt_texts


class DataLoader:
    """
    Loads the IAM-Dataset. Based on the DataLoader from SimpleHTR,
    https://github.com/githubharald/SimpleHTR
    """

    def __init__(self, file_path, batch_size, img_size, max_text_len):
        """
        Loader for the dataset at the given location. Preprocesses images and
        text according to the parameters
        :param file_path: Path to the data
        :param batch_size: Batch size
        :param img_size: size of the image
        :param max_text_len: the maximum text label length
        """
        assert file_path[-1] == '/'
        self.data_augmentation = False
        self.current_index = 0
        self.batch_size = batch_size
        self.img_size = img_size
        self.samples = []

        f = open(file_path + 'words.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']

        for line in f:
            # Ignore the comment line
            if not line or line[0] == '#':
                continue
            line_split = line.strip().split(' ')
            assert len(line_split) >= 9
            file_name_split = line_split[0].split('-')
            file_name = file_path + 'words/' + file_name_split[0] + '/' + \
                file_name_split[0] + '-' + file_name_split[1] + '/' + \
                line_split[0] + '.png'

            # GT text are columns starting at position 9
            gt_text = self.truncate_label(' '.join(line_split[8:]),
                                          max_text_len)
            chars = chars.union(set(list(gt_text)))

            # check if image is not empty
            if not os.path.getsize(file_name):
                bad_samples.append((line_split[0] + '.png'))
                continue
            self.samples.append(Sample(gt_text, file_name))
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set
        split_index = int(0.95 * len(self.samples))
        self.train_samples = self.samples[:split_index]
        self.validation_samples = self.samples[split_index:]

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # number of randomly chosen samples per epoch for training
        self.number_samples_per_epoch = 25000
        self.train_set()

        # list of all chars in the dataset
        self.char_list = sorted(list(chars))

    def truncate_label(self, text, max_text_len):
        """
        adds cost for repeat letters and cuts the text label if the cost ist
        higher than the max_text_len. ctc_loss can't compute the loss if it
        cannot find a mapping between text label and input labels. If a too-
        long label is provided, ctc_loss returns an infinite gradient.
        :param text: The wordlabel from words.txt
        :param max_text_len: the maximum text label length
        :return: the text label
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def train_set(self):
        """
        switches to a randomly chosen subset of the training set.
        :return: nothing, stores the chosen samples in self.samples
        """
        self.data_augmentation = True
        self.current_index = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples[:self.number_samples_per_epoch]

    def validation_set(self):
        """
        switches to the validation set
        :return: nothing, stores the validation samples in self.samples
        """
        self.data_augmentation = False
        self.current_index = 0
        self.samples = self.validation_samples

    def get_iterator_info(self):
        """
        :return: current batch index and overall number of batches
        """
        return (self.current_index // self.batch_size + 1, len(self.samples) //
                self.batch_size)

    def has_next(self):
        """
        :return: True if the end of the epoch is not yet reached
        """
        return self.current_index + self.batch_size <= len(self.samples)

    def get_next(self):
        """
        Gets the next batch.
        :return: a batch object with a list of textlabels and images.
        """
        batch_range = range(self.current_index, self.current_index +
                            self.batch_size)
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        images = [preprocess(cv2.imread(self.samples[i].file_path,
                                        cv2.IMREAD_GRAYSCALE), self.img_size,
                             self.data_augmentation) for i in batch_range]
        self.current_index += self.batch_size
        return Batch(gt_texts, images)
