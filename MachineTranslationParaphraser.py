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
        self.tokenizer = '' # TODO add tokenizer

    def _load_translators(self, forward_model_path='transformer.wmt19.en-de.single_model',
                          backward_model_path='transformer.wmt19.de-en.single_model'):
        forward = torch.hub.load('pytorch/fairseq', forward_model_path, tokenizer='moses', bpe='fastbpe')
        backward = torch.hub.load('pytorch/fairseq', backward_model_path, tokenizer='moses', bpe='fastbpe')

        return forward.eval(), backward.eval()

    def backtranslate(self, sent):
        return self.backward_model.translate(self.forward_model.translate(sent))

    def _generate_paraphrases(
            self,
            input_text: str,
            max_length: int,
            **kwargs
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Paraphrases the input text.
        :param input_text: the text to paraphrase. Should be a single string. Example:
            '''Around midnight at London's Leicester Square, as news of Jackson's death spread, Luis Carlos Ameida and his friends were surrounding a car listening to the star's music. Ameida said he'd gotten tickets to see Jackson at his "This Is It" concerts beginning on July 13 in London.'''
        :param max_length: the maximum length of the paraphrase.
        :return: a list of paraphrased strings
        """
        # run input text through backtranslate
        forward_translation = self.forward_model.translate(input_text, beam=10)
        
        # backward translate
        forward_tokens = self.forward_model.tokenize(forward_translation)
        forward_bpe = self.forward_model.apply_bpe(forward_tokens)
        forward_bin = self.forward_model.binarize(forward_bpe)
        paraphrases_bin_list = self.backward_model.generate(forward_bin, beam=10, sampling=True, sampling_topk=20)
        
        # detokenise and create return lists
        paraphrase_list = []
        paraphrase_tokens_list = []
        for paraphrase_bin in paraphrases_bin_list:
            paraphrase_sample = paraphrase_bin['tokens']
            paraphrase_bpe = self.backward_model.string(paraphrase_sample)
            paraphrase_tokens = self.backward_model.remove_bpe(paraphrase_bpe)
            paraphrase_string = self.backward_model.detokenize(paraphrase_tokens)
            
            paraphrase_list.append(paraphrase_string)
            paraphrase_tokens_list.append(paraphrase_tokens)
            
        
        #return raw output and tokenized input 
        return paraphrase_list, paraphrase_tokens_list
