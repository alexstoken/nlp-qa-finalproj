"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch
from typing import *
from torch.utils.data import Dataset
from random import shuffle
import numpy as np
from utils import *
from AbstractParaphraser import AbstractParaphraser

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """

    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for ex in samples:
            passage_tokens = ex[PASSAGE_TOKENS_KEY]
            question_tokens = ex[QUESTION_TOKENS_KEY]
            for token in itertools.chain(passage_tokens, question_tokens):
                vocab[token.lower()] += 1
        top_words = [word for (word, _) in sorted(vocab.items(), key=lambda x: x[1], reverse=True)][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words

    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """

    def __init__(self, args, paths):
        
        self.args = args
        elems = []

        # most dev QA datasets will only be one path, so handle that case
        if type(paths)!= list:
            paths= [paths]
            
        for path in paths:
            _, elem = load_dataset(path)
            elems.extend(elem)
        
        self.samples: List[Dict] = self._create_examples(
            elems,
            num_answers=args.num_answers,
            lowercase_passage=args.lowercase_passage,
            lowercase_question=args.lowercase_question,
            max_context_length=args.max_context_length,
            max_question_length=args.max_question_length,
            ngram = args.ngram,
            paraphrase_score_thresh= args.paraphrase_score_thresh,
            paraphrase_sampling_rate= args.paraphrase_sampling_rate
        )
        self.num_paraphrased_questions: int = self._calculate_num_paraphrased_questions(elems)
        self.tokenizer = None
        self.batch_size: int = args.batch_size
        self.pad_token_id: int = self.tokenizer.pad_token_id if (self.tokenizer is not None) else 0

    @classmethod
    def _create_examples(
            cls,
            elems: List[Dict],
            num_answers: int,
            lowercase_passage: bool,
            lowercase_question: bool,
            max_context_length: int,
            max_question_length: int,
            ngram: int,
            paraphrase_score_thresh: float = 0.,
            paraphrase_sampling_rate: float = 1.,
    ) -> List[Dict]:
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        max_num_answers = 0
        examples: List[Dict] = []
        for elem in elems:
            
            # Unpack the context paragraph (passage). Shorten to max sequence length.
            passage_tokens_idxs: List[Tuple[str, int]] = elem['context_tokens']
            passage_tokens: List[str] = [
                token.lower() if lowercase_passage else token
                for (token, offset) in passage_tokens_idxs[:max_context_length]
            ]
            ## Get the passage string and shorten it to to max sequence length.
            passage: str = elem['context']
            final_passage_token_idx: int = min(max_context_length, len(passage_tokens_idxs)) - 1
            final_passage_token__start_idx: int = passage_tokens_idxs[final_passage_token_idx][1]
            final_passage_token_len = len(passage_tokens_idxs[final_passage_token_idx][0])
            final_passage_idx = final_passage_token__start_idx + final_passage_token_len
            passage: str = passage[:final_passage_idx]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:

                qid = qa['qid']
                
                # all paraphrase datasets have this extra key
                # if paraphrase dataset, choose whether to get paraphrase or not
                if 'question_is_paraphrased' in qa:
        
                    
                    # if the question isn't changed at all, skip
                    if qa['question'] == qa['question_original']:
                        continue
                        
                    # calcualte paraphrase score
                    question_tokens_original = [x[0] for x in qa['question_tokens_original']]
                    question_tokens = [x[0] for x in qa['question_tokens']]
                    
                    paraphrase_score = AbstractParaphraser._calculate_paraphrase_score(question_tokens_original, question_tokens, ngram)
                    # if score is above threshold, use paraphrase sample with some probability (sampling rate)
                    if paraphrase_score > paraphrase_score_thresh and np.random.rand() < paraphrase_sampling_rate:
                        question_tokens_idxs: List[Tuple[str, int]] = qa['question_tokens']
                        question: str = qa['question']
                    # otherwise use original tokens
                    else:
                        question_tokens_idxs: List[Tuple[str, int]] = qa['question_tokens_original']
                        question: str = qa['question_original']
                            
                # if a dataset doesnt have paraphrases in it, we always grab question and question tokens
                else:
                    question_tokens_idxs: List[Tuple[str, int]] = qa['question_tokens']
                    question: str = qa['question']
                        
                # additional processing of question
                question_tokens: List[str] = [
                    token.lower() if lowercase_question else token
                    for (token, offset) in question_tokens_idxs[:max_question_length]
                ]
                
                final_question_token_idx: int = min(max_question_length, len(question_tokens_idxs)) - 1
                final_question_token_start_idx: int = question_tokens_idxs[final_question_token_idx][1]
                final_question_token_len = len(question_tokens_idxs[final_question_token_idx][0])
                final_question_idx = final_question_token_start_idx + final_question_token_len
                question: str = question[:final_question_idx]

                # Select one or more answer spans, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answer_examples = []
                for answer in qa['detected_answers']:
                    answer_start_idx, answer_end_idx = answer['token_spans'][0]
                    answer_examples.append({
                        QID_KEY: qid,
                        PASSAGE_TOKENS_KEY: passage_tokens,
                        PASSAGE_KEY: passage,
                        QUESTION_TOKENS_KEY: question_tokens,
                        QUESTION_KEY: question,
                        ANSWER_START_IDX_KEY: answer_start_idx,
                        ANSWER_END_IDX_KEY: answer_end_idx
                    })
                max_num_answers = max(max_num_answers, len(answer_examples))
                if num_answers >= 1:
                    answer_examples = answer_examples[:num_answers]
                    # print(f'Restricting from {len(answer_examples)} to {num_answers} answers')
                examples += answer_examples
        print(f'Max number of answers in this dataset: {max_num_answers}')
        return examples

    @classmethod
    def _calculate_num_paraphrased_questions(cls, elems: List[Dict]) -> int:
        num_paraphrased_questions = 0
        for example in elems:
            for qa_dict in example['qas']:
                if 'question_original' in qa_dict and qa_dict['question'] != qa_dict.get('question_original'):
                    num_paraphrased_questions += 1
        return num_paraphrased_questions

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        start_positions = []
        end_positions = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid = self.samples[idx]['qid']
            passage_tokens = self.samples[idx]['passage_tokens']
            question_tokens = self.samples[idx]['question_tokens']
            answer_start_idx = self.samples[idx]['answer_start_idx']
            answer_end_idx = self.samples[idx]['answer_end_idx']

            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(passage_tokens)
            )
            question_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(question_tokens)
            )
            answer_start_ids = torch.tensor(answer_start_idx)
            answer_end_ids = torch.tensor(answer_end_idx)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)

        return zip(passages, questions, start_positions, end_positions)

    def _create_batches(self, generator, batch_size) -> Dict:
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages = torch.zeros(bsz, max_passage_length)
            padded_questions = torch.zeros(bsz, max_question_length)
            # Pad passages and questions
            for iii, passage_question in enumerate(zip(passages, questions)):
                passage, question = passage_question
                padded_passages[iii][:len(passage)] = passage
                padded_questions[iii][:len(question)] = question

            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long()
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)
