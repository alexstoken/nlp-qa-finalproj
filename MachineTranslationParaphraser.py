from typing import *
from utils import *
import ast, logging, tensorboard as tb, os, re, argparse, json, io, copy
from abc import ABC, abstractmethod
from collections import Counter
import nltk
from pprint import pprint
from AbstractParaphraser import AbstractParaphraser


class MachineTranslationParaphraser(AbstractParaphraser):
    pass


class FairSeqParaphraser(MachineTranslationParaphraser):
    PRETRAINED_MODEL_NAMES = torch.hub.list('pytorch/fairseq')  # grab list from torch hub

    def __init__(self, args, device):
        super().__init__(args, device)
        self.forward_model, self.backward_model = self._load_translators()

    def _load_translators(self, forward_model_path='transformer.wmt19.en-de.single_model',
                          backward_model_path='transformer.wmt19.de-en.single_model'):
        forward = torch.hub.load('pytorch/fairseq', forward_model_path, tokenizer='moses', bpe='fastbpe')
        backward = torch.hub.load('pytorch/fairseq', backward_model_path, tokenizer='moses', bpe='fastbpe')

        return forward.eval(), backward.eval()

    def backtranslate(self, sent):
        return self.backward_model.translate(self.forward_model.translate(sent))

    def _generate_paraphrases(self):
        return

    pass
