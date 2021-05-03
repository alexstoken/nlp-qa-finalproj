from typing import *
from utils import *
import ast, logging, tensorboard as tb, os, re, argparse, json, io, copy
from abc import ABC, abstractmethod
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from collections import Counter
import nltk
from pprint import pprint
from AbstractParaphraser import AbstractParaphraser


class AbstractiveSummarizationParaphraser(AbstractParaphraser):
    pass


class PegasusParaphraser(AbstractiveSummarizationParaphraser):
    GENERATED_NEWLINE = '<n>'

    ## Found on HuggingFace search bar, from the PEGASUS paper released by Google: https://arxiv.org/pdf/1912.08777.pdf
    PRETRAINED_MODEL_NAMES = [
        "google/pegasus-large",
        "google/pegasus-xsum",
        "google/pegasus-cnn_dailymail",
        "google/pegasus-multi_news",
    ]

    def __init__(self, args, device):
        super().__init__(args, device)
        assert args.pretrained_model_name in self.PRETRAINED_MODEL_NAMES
        self.pretrained_model_name = args.pretrained_model_name
        logging.info(f'Loading Pegasus ({self.pretrained_model_name})')
        self.model = PegasusForConditionalGeneration.from_pretrained(self.pretrained_model_name).to(self.device)
        self.tokenizer: PegasusTokenizer = PegasusTokenizer.from_pretrained(self.pretrained_model_name)

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
        assert isinstance(input_text, str)

        tokenized_ids = self.tokenizer(
            [input_text],
            truncation=True,
            padding='longest',
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        paraphrased_token_ids_list = self.model.generate(
            **tokenized_ids,
            max_length=max_length,
            num_beams=self.args.num_beams,
            num_return_sequences=self.args.num_beams,
            temperature=self.args.temperature,
        )
        paraphrases_list: List[str] = [
            paraphrase.replace(self.GENERATED_NEWLINE, ' ')
            for paraphrase in self.tokenizer.batch_decode(paraphrased_token_ids_list, skip_special_tokens=True)
        ]
        paraphrase_tokens_list: List[List[str]] = [
            self.tokenize(paraphrase) for paraphrase in paraphrases_list
        ]
        # paraphrase_tokens_list: List[List[str]] = [
        #     self.undo_sentencepiece_tokenization(
        #         self.tokenizer.convert_ids_to_tokens(paraphrased_token_ids, skip_special_tokens=True)
        #     )
        #     for paraphrased_token_ids in paraphrased_token_ids_list
        # ]
        del tokenized_ids, paraphrased_token_ids_list
        return paraphrases_list, paraphrase_tokens_list

    # @classmethod
    # def undo_sentencepiece_tokenization(cls, subword_tokens: List[str]) -> List[str]:
    #     """
    #     Undo the tokenization.
    #     :param subword_tokens:
    #     Example#1:
    #         ['▁Luis', '▁Carlos', '▁A', 'me', 'ida', '▁said', '▁he', "'", 'd', '▁gotten', '▁tickets', '▁to', '▁see', '▁Jackson', '▁at', '▁his', '▁"', 'This', '▁I', 's', '▁It', '"', '▁concerts', '▁', '.', '<n>', 'A', 'me', 'ida', '▁said', '▁he', "'", 'd', '▁gotten', '▁tickets', '▁to', '▁see', '▁Jackson', '▁at', '▁his', '▁"', 'This', '▁I', 's', '▁It', '"', '▁concerts', '▁beginning', '▁on', '▁July', '▁13', '▁in', '▁London', '▁', '.']
    #     Example#2:
    #         ['▁Peter', '▁Mai', 'y', 'oh', '▁is', '▁a', '▁Kenyan', '▁student', '▁studying', '▁in', '▁the', '▁U', '.', 'S', '.', '▁city', '▁of', '▁Kansas', ',', '▁Missouri', '▁', '.', '<n>', 'He', '▁says', '▁Jackson', '▁"', 'was', '▁there', '▁before', '▁Tiger', '▁Woods', ',', '▁before', '▁Michael', '▁Jordan', ',', '▁even', '▁before', '▁Barack', '▁Obama', '"', '<n>', 'Mai', 'y', 'oh', ':', '▁"', 'I', '▁hope', '▁people', '▁remember', '▁him', '▁for', '▁the', '▁work', '▁he', '▁did', '"']
    #     :return:
    #     """
    #     tokens = []
    #     ## The character ▁ (NOT an underscore) is used by sentencepiece to represent a space.
    #     ## Ref: https://huggingface.co/transformers/tokenizer_summary.html#sentencepiece
    #     WORD_START_CHAR = '▁'
    #     NEWLINE_CHAR = '<n>'
    #     clean = lambda x: x.replace(WORD_START_CHAR, '').replace('<n>', ' ')
    #     for subword_token in subword_tokens:
    #         if len(tokens) == 0:
    #             tokens.append(clean(subword_token))
    #         else:
    #             if subword_token == WORD_START_CHAR:
    #
    #             if subword_token.startswith(WORD_START_CHAR):
    #                 tokens.append(clean(subword_token))
    #
    #     [
    #
    #         token.replace('▁', '')
    #         ## Example: ['▁Luis', '▁Carlos', '▁A', 'me', 'ida', '▁said', '▁he', "'", 'd', '▁gotten', '▁tickets', '▁to', '▁see', '▁Jackson', '▁at', '▁his', '▁"', 'This', '▁I', 's', '▁It', '"', '▁concerts', '▁', '.', '<n>', 'A', 'me', 'ida', '▁said', '▁he', "'", 'd', '▁gotten', '▁tickets', '▁to', '▁see', '▁Jackson', '▁at', '▁his', '▁"', 'This', '▁I', 's', '▁It', '"', '▁concerts', '▁beginning', '▁on', '▁July', '▁13', '▁in', '▁London', '▁', '.']
    #         for token in
    #         if token != self.GENERATED_NEWLINE
    #     ]
