from typing import *
from utils import *
import ast, logging, tensorboard as tb, os, re, argparse, json, io, copy
from abc import ABC, abstractmethod
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from collections import Counter
import nltk
import numpy as np
from pprint import pprint

nltk.download('punkt')
from nltk.tokenize import word_tokenize


class AbstractParaphraser(ABC):
    ## Stopwords are taken from NLTK:
    PUNCTUATION = {'.', ',', ';', ':', '-', '--'}
    STOPWORDS = {
        'ourselves', 'hers', 'between', 'yourself', 'but', 'again',
        'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own',
        'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most',
        'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
        'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through',
        'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
        'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no',
        'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will',
        'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so',
        'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where',
        'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
        'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
        'here', 'than'
    }

    def __init__(self, args, device):
        self.args = args
        self.device = device

    @classmethod
    def tokenize(cls, string: str) -> List[str]:
        return [x.replace('``', '"') for x in word_tokenize(string)]

    def paraphrase(
            self,
            passage: str,
            passage_tokens: List[Tuple[str, int]],
    ) -> Tuple[str, List[str]]:
        ## Driver function
        start = time.time()
        text_chunks, text_chunks_tokens_list = self._chunk_passage(
            passage,
            normalized_chunk_length=self.args.normalized_chunk_length
        )
        # print(f'text_chunks: {text_chunks}')
        # print(f'text_chunks_tokens_list: {text_chunks_tokens_list}')
        paraphrase_chunks: List[str] = []
        paraphrase_tokens: List[str] = []
        logging.info(f'\nDivided input into {len(text_chunks)} chunks...')
        for text_chunk, text_chunk_tokens in zip(text_chunks, text_chunks_tokens_list):
            paraphrases_list, paraphrase_tokens_list = self._generate_paraphrases(
                text_chunk,
                max_length=int(round(self.args.max_len_multiplier * len(text_chunk.split(' ')))),
            )
            paraphrase_chunk, paraphrase_tokens_chunk = self._select_paraphrase(
                passage=text_chunk,
                passage_tokens=text_chunk_tokens,
                paraphrases_list=paraphrases_list,
                paraphrase_tokens_list=paraphrase_tokens_list,
                use_scoring_function=self.args.use_scoring_function,
                score_threshold=self.args.score_threshold,
                score_n_gram=self.args.score_n_gram,
                print_top_k=self.args.print_top_k_paraphrases,
                print_bottom_k=self.args.print_bottom_k_paraphrases,
            )
            if (paraphrase_chunk, paraphrase_tokens_chunk) == (None, None):
                paraphrase_chunk, paraphrase_tokens_chunk = text_chunk, [x[0] for x in passage_tokens]
            paraphrase_chunks.append(paraphrase_chunk)
            paraphrase_tokens += paraphrase_tokens_chunk
        paraphrase: str = self.args.paraphrase_joiner.join(paraphrase_chunks)
        end = time.time()
        logging.info(
            f'Took {end - start:.3f} seconds to paraphrase text of length {len(passage_tokens)} tokens, '
            f'(as {len(text_chunks)} chunks each of length approx. {self.args.normalized_chunk_length} tokens)'
        )
        return paraphrase, paraphrase_tokens

    '''
        Example `passage_tokens`:
        ['(', 'OPRAH.com', ')', '--', 'He', 'can', 'saw', 'himself', 'in', 'half', ',', 'sing', 'a', 'selection', 'of',
         'Broadway', 'showtunes', 'and', 'swing', 'on', 'a', 'flying', 'trapeze', '.', 'Neil', 'Patrick', 'Harris',
         'says', 'he', "'ll", 'try', 'to', 'make', 'viewers', 'feel', 'like', 'they', "'re", 'in', 'good', 'hands',
         'with', 'him', 'as', 'Emmy', 'host', '.', 'When', 'Neil', 'Patrick', 'Harris', ',', 'one', 'of', 'the',
         'stars',
         'of', 'the', 'hit', 'CBS', 'sitcom', '"', 'How', 'I', 'Met', 'Your', 'Mother', ',', '"', 'is', "n't",
         'dabbling', 'in', 'the', 'extraordinary', ',', 'well', ',', 'he', "'s", 'probably', 'hosting', 'an', 'awards',
         'show', '.', 'In', 'the', 'late', "'", '80s', ',', 'Neil', '--', 'known', 'as', 'NPH', 'to', 'his', 'fans',
         '--', 'landed', 'the', 'starring', 'role', 'on', '"', 'Doogie', 'Howser', ',', 'M.D.', '"', 'After', 'years',
         'of', 'child', 'stardom', 'and', 'teen', 'heartthrob', 'status', ',', 'Neil', 'left', 'the', 'small', 'screen',
         'for', 'the', 'stage', '.', 'He', 'became', 'a', 'respected', 'Broadway', 'actor', ',', 'starring', 'in',
         'shows', 'like', '"', 'Rent', ',', '"', '"', 'Cabaret', '"', 'and', '"', 'Proof', ',', '"', 'before',
         'returning', 'to', 'television', '.', 'Now', ',', 'millions', 'know', 'Neil', 'as', 'Barney', 'Stinson', ',',
         'the', 'womanizing', ',', 'slap', '-', 'happy', 'sidekick', 'on', '"', 'How', 'I', 'Met', 'Your', 'Mother',
         ',', '"', 'which', 'begins', 'its', 'fifth', 'season', 'September', '21', '.', 'Like', 'Billy', 'Crystal',
         'and', 'Johnny', 'Carson', 'before', 'him', ',', 'this', 'man', '-', 'of', '-', 'many', '-', 'talents', 'is',
         'also', 'making', 'his', 'mark', 'as', 'an', 'awards', 'show', 'host', '.', 'On', 'Sunday', ',', 'September',
         '20', ',', 'Neil', 'will', 'host', 'the', '61st', 'Primetime', 'Emmy', 'Awards', '.', 'He', 'shares', 'his',
         'thoughts', 'on', 'fate', ',', 'finding', 'balance', 'and', 'making', 'out', 'with', 'his', 'co', '-', 'star',
         '.', 'Kari', 'Forsee', ':', 'How', 'are', 'you', 'preparing', 'for', 'Emmy', 'night', '?', 'Neil', 'Patrick',
         'Harris', ':', 'I', "'m", 'just', 'trying', 'to', 'make', 'sure', 'all', 'the', 'comedy', 'host', 'elements',
         'are', 'in', 'place', '.', 'We', "'ll", 'have', 'a', 'good', 'opening', 'bit', 'and', 'a', 'couple',
         'surprise', 'things', 'throughout', '.', 'We', 'want', 'to', 'balance', 'respecting', 'the', 'show', 'and',
         'the', 'doling', 'out', 'of', 'the', 'awards', 'with', 'the', 'sort', 'of', 'random', 'things', 'that', 'will',
         'keep', 'the', 'audience', "'s", 'attention', 'in', 'other', 'ways', '.', 'So', 'that', "'s", 'kind', 'of',
         'been', 'my', 'job', '.', 'You', 'want', 'to', 'make', 'it', 'unique', 'and', ',', 'yet', ',', 'classic', '.',
         'That', "'s", 'a', 'tricky', 'dynamic', '.', 'Oprah.com', ':', 'Planning', 'an', 'Emmys', 'party', '?', 'Get',
         '4', 'entertaining', 'solutions', 'KF', ':', 'I', 'can', 'imagine', '.', 'How', 'often', 'are', 'you',
         'rehearsing', '?', 'NPH', ':', 'Well', ',', 'it', "'s", 'sort', 'of', 'a', 'litany', 'of', 'e', '-', 'mails',
         'and', 'phone', 'calls', 'all', 'day', 'with', 'the', 'producers', '.', 'We', 'had', 'a', 'great', 'opening',
         'short', 'film', 'we', 'are', 'going', 'to', 'shoot', ',', 'and', 'it', 'would', 'be', 'the', 'first', 'thing',
         'you', 'shot', '.', 'That', 'was', 'going', 'to', 'be', 'with', 'Alec', 'Baldwin', ',', 'and', 'he',
         'withdrew', 'at', 'the', 'last', 'minute', '.', 'So', 'that', 'got', 'scrapped', ',', 'and', 'we', "'re",
         'off', 'to', 'plan', 'D', ',', 'E', 'or', 'F.', 'It', "'s", 'sort', 'of', 'like', 'now', 'you', 'go', ':', '"',
         'That', "'s", 'fantastic', ',', 'great', '.', 'We', "'ve", 'got', 'that', 'person', ',', '"', 'or', '"', 'Oh',
         ',', 'that', 'person', 'did', "n't", 'work', '.', 'Now', 'what', 'do', 'we', 'do', '?', '"', 'A', 'lot', 'of',
         '"', 'now', 'what', 'do', 'we', 'do', '?', '"', 'questions', '.', 'KF', ':', 'Now', 'at', 'the', 'Tony',
         'Awards', ',', 'you', 'sang', 'a', ',', 'may', 'I', 'say', ',', 'legendary', 'closing', 'number', '.', 'Will',
         'you', 'be', 'singing', 'at', 'the', 'Emmys', ',', 'or', 'is', 'dancing', 'more', 'the', 'focus', '?', 'NPH',
         ':', 'I', 'suspect', 'you', 'wo', "n't", 'see', 'me', 'dancing', 'very', 'much', '.', 'That', "'s", 'not',
         'my', 'forte', '.', 'But', 'yeah', ',', 'I', 'might', 'throw', 'some', 'sort', 'of', 'singing', 'into', 'it',
         '.', 'I', 'have', "n't", 'quite', 'decided', '.', 'I', 'sort', 'of', 'feel', 'like', 'the', 'Emmys', 'are',
         'so', 'classy', 'and', 'glamorous', 'and', 'black', 'tie', ',', 'the', 'host', 'really', 'needs', 'to',
         'respect', 'his', 'job', 'title', '.', 'I', 'think', 'too', 'much', '"', 'Look', 'at', 'me', '!', 'Look', 'at',
         'me', '!', '"', 'as', 'the', 'host', 'of', 'a', 'show', 'that', 'big', 'is', 'counterproductive', '.', 'So',
         'long', 'as', 'I', 'make', 'you', 'feel', 'confident', 'that', 'you', "'re", 'in', 'good', 'hands', 'with',
         'me', 'as', 'the', 'host', ',', 'then', 'it', "'s", 'my', 'real', 'responsibility', 'to', 'introduce', 'you',
         'to', 'a', 'lot', 'of', 'other', 'people', 'and', 'elements', '--', 'other', 'presenters', 'who', 'are',
         'then', 'going', 'to', 'talk', 'to', 'you', 'or', 'other', 'introductions', 'of', 'next', 'sections', '.',
         'That', "'s", 'my', 'role', '.', 'It', "'s", 'not', 'really', 'to', 'be', 'a', 'song', '-', 'and', '-',
         'dance', 'man', '.', 'KF', ':', 'Did', 'you', 'look', 'back', 'at', 'past', 'Emmy', 'hosts', 'for',
         'inspiration', '?', 'NPH', ':', 'Very', 'much', '.', 'Steve', 'Allen', 'hosted', 'the', 'first', 'televised',
         'awards', ',', 'which', 'was', 'the', '7th', 'Annual', 'Emmy', 'Awards', ',', 'in', ',', 'I', 'think', ',',
         '1955', ',', 'and', 'he', 'was', 'great', '.', 'That', 'was', 'sort', 'of', 'my', 'inspiration', 'for', 'all',
         'of', 'this', '.', 'He', 'just', 'had', 'such', 'a', 'dry', 'wit', ',', 'a', 'commanding', 'voice', ',', 'a',
         'great', 'presence', '.', 'You', 'knew', 'when', 'you', 'were']
    '''
    '''
        Example `qa`:
        {
            "id": "./cnn/stories/8b66f5754a70c62dbacda0e7ab9d1a369b0b6f2e.story#0",
            "qid": "fd58ed81f9ea41e2b803c29dc3d424d0",
            "question": "What character does he play on \"How I Met Your Mother\"?",
            "question_tokens": [
                ["What", 0], 
                ["character", 5], 
                ["does", 15], 
                ["he", 20], 
                ["play", 23], 
                ["on", 28], 
                ["\"", 31],  
                ["How", 32],  
                ["I", 36],  
                ["Met", 38],  
                ["Your", 42],  
                ["Mother", 47],  
                ["\"", 53], 
                ["?",54]
            ],
            "answers": [
                "Barney Stinson,"
            ],
            "detected_answers": [
                {
                    "text": "Barney Stinson,",
                    "char_spans": [[756,770]],
                    "token_spans": [[165,167]]
                }
            ]
        }
    '''
    @classmethod
    def get_token_idxs(cls, string: str, tokens: List[str]) -> List[Tuple[str, int]]:
        tokens_idxs = []
        # print()
        # print(string)
        # print(tokens)
        prev_token_idx = 0
        for token in tokens:
            idx = string.find(token, prev_token_idx)
            if idx == -1:
                idx = prev_token_idx
            tokens_idxs.append([token, idx])
            prev_token_idx = idx + 1
        assert len(tokens_idxs) == len(tokens)
        return tokens_idxs


    def paraphrase_question(self, example: Dict) -> Dict:
        """
        Paraphrapse the question using this model.
        :param example: one entry in the dataset. A dict with the keys 'context', 'context_tokens', 'qas', etc.
        :return: example, with paraphrased question.
        """
        example = copy.deepcopy(example)
        passage: str = example['context']
        passage_tokens: List[str] = [x[0] for x in example['context_tokens']]
        for qa_dict in example['qas']:
            question: str = qa_dict['question']
            question_tokens: List[Tuple[str, int]] = qa_dict['question_tokens']
            qa_dict['question_original']: str = question
            qa_dict['question_tokens_original']: List[Tuple[str, int]] = question_tokens
            paraphrased_question, paraphrased_question_tokens = self.paraphrase(question, question_tokens)
            qa_dict['question']: str = paraphrased_question
            qa_dict['question_tokens']: List[Tuple[str, int]] = self.get_token_idxs(
                paraphrased_question, paraphrased_question_tokens
            )
            qa_dict['question_is_paraphrased']: bool = True
            # for answer_dict in qa_dict['detected_answers']:
            #     passage_tokens_ans_start_idx = answer_dict['token_spans'][0][0]
            #     passage_tokens_ans_end_idx = answer_dict['token_spans'][0][1] + 1
            #     ## Example `answer_tokens`:   ['Barney', 'Stinson', ',']
            #     answer_tokens: List[str] = passage_tokens[passage_tokens_ans_start_idx:passage_tokens_ans_end_idx]
            #     ## Example `answer`:   "Barney Stinson,"
            #     answer: str = answer_dict['text']
            #     passage_ans_start_idx = answer_dict['char_spans'][0][0]
            #     passage_ans_end_idx = answer_dict['char_spans'][0][1] + 1
            #     assert answer == passage[passage_ans_start_idx:passage_ans_end_idx]
            #     assert isinstance(answer_tokens, list)
        return example

    def paraphrase_around_answer_sent(self, example: Dict) -> Dict:
        """
        Paraphrapse the answer using this model.
        :param example: one entry in the dataset. A dict with the keys 'context', 'context_tokens', 'qas', etc.
        :return: example, with paraphrased passage EXCEPT for the answer sentence.
        """
        pass  ##TODO

    def paraphrase_answer(self, example: Dict) -> Dict:
        """
        Paraphrapse the answer using this model.
        :param example: one entry in the dataset. A dict with the keys 'context', 'context_tokens', 'qas', etc.
        :return: example, with paraphrased answer.
        """
        pass  ##TODO

    def paraphrase_answer_sent(self, example: Dict) -> Dict:
        """
        Paraphrapse the answer sentence using this model.
        :param example: one entry in the dataset. A dict with the keys 'context', 'context_tokens', 'qas', etc.
        :return: example, with paraphrased answer sentence.
        """
        pass  ##TODO

    @abstractmethod
    def _generate_paraphrases(self, *args, **kwargs) -> Tuple[List[str], List[List[str]]]:
        pass

    def _select_paraphrase(
            self,
            passage: str,
            passage_tokens: List[str],
            paraphrases_list: List[str],
            paraphrase_tokens_list: List[List[str]],
            use_scoring_function: bool,
            score_threshold: float,
            score_n_gram: int,
            print_top_k: int,
            print_bottom_k: int,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        if use_scoring_function:
            scored_paraphrases: List[Tuple[float, str, List[str]]] = self._calculate_paraphrase_scores(
                passage_tokens=passage_tokens,
                paraphrases_list=paraphrases_list,
                paraphrase_tokens_list=paraphrase_tokens_list,
                n_gram=score_n_gram,
            )
            scored_paraphrases: List[Tuple[float, str, List[str]]] = sorted(
                scored_paraphrases, key=lambda x: x[0], reverse=True
            )
            if print_top_k > 0 or print_bottom_k > 0:
                logging.info('Paraphrasing...\n')
                logging.info(f'> Original:\n{passage}\n')
            if print_top_k > 0:
                logging.info(f'> Top {print_top_k} paraphrases by scores (n_gram={score_n_gram}):')
                for paraphrase_score, paraphrase, paraphrase_tokens in scored_paraphrases[:print_top_k]:
                    logging.info((paraphrase_score, paraphrase))
                    logging.info('')
            if print_bottom_k > 0:
                logging.info(f'> Bottom {print_top_k} paraphrases by scores (n_gram={score_n_gram}):')
                for paraphrase_score, paraphrase, paraphrase_tokens in scored_paraphrases[-print_bottom_k:]:
                    logging.info((paraphrase_score, paraphrase))
                    logging.info('')
            paraphrase_score, paraphrase, paraphrase_tokens = scored_paraphrases[0]  ## Select those with highest score
            if paraphrase_score >= score_threshold:
                return paraphrase, paraphrase_tokens
            else:
                ## Use original passage and its tokens
                return None, None
        else:
            ## Return the one with the highest beam search score:
            scored_paraphrases = [
                (i + 1, paraphrase, paraphrase_tokens)  ## First element is paraphrase rank
                for i, (paraphrase, paraphrase_tokens) in enumerate(zip(paraphrases_list, paraphrase_tokens_list))
            ]
            if print_top_k > 0 or print_bottom_k > 0:
                logging.info('Paraphrasing...\n')
                logging.info(f'> Original:\n{passage}\n')
            if print_top_k > 0:
                logging.info(f'> Top {print_top_k} paraphrases by beam search:')
                for paraphrase_rank, paraphrase, paraphrase_tokens in scored_paraphrases[:print_top_k]:
                    logging.info((paraphrase_rank, paraphrase), end='\n\n')
            if print_bottom_k > 0:
                logging.info(f'> Bottom {print_top_k} paraphrases by beam search:')
                for paraphrase_rank, paraphrase, paraphrase_tokens in scored_paraphrases[-print_bottom_k:]:
                    logging.info((paraphrase_rank, paraphrase), end='\n\n')
            ## Return those with highest rank:
            return paraphrases_list[0], paraphrase_tokens_list[0]

    @classmethod
    def _chunk_passage(
            cls,
            passage: str,
            normalized_chunk_length: int,
            token_sep: str = ' ',
            line_sep: str = '.'
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Chunk the passage before feeding it into the AbstractParaphraser.
        :param passage: string of the passage.
        :param passage_tokens: List of strings of the passage tokens.
        :param normalized_chunk_length: the normalized length of any one chunk. We will try to split/combine adjacent
            sentences to make it close to this length.
        :return:
        """
        passage = re.sub('(\n)+', '\n', passage)  ## Replace multiple newlines with a single newline.
        passage = re.sub('( )+', ' ', passage.strip())  ## Delete redundant spaces.
        assert isinstance(passage, str) and len(passage) > 0
        passage_chunks: List[str] = passage.split('\n')

        num_tokens = lambda x: len(x.split(token_sep))

        ## Subdivide large chunks into smaller chunks and fold them back into the existing list of chunks:
        passage_chunks_split: List[str] = []
        for passage_chunk in passage_chunks:
            if num_tokens(passage_chunk) < normalized_chunk_length:
                passage_chunks_split.append(passage_chunk)
            else:
                ## Create a new list of sub-chunks for the current chunk. By default, each subchunk is a line.
                passage_subchunks = passage_chunk.split(line_sep + token_sep)
                ## The sub-chunks might be short. Keep concatenating them to form longer sub-chunks.
                passage_subchunk_concat = []
                for subchunk in passage_subchunks:
                    if not subchunk.endswith(line_sep):
                        subchunk += line_sep
                    ## Add the very first sub-chunk:
                    if len(passage_subchunk_concat) == 0:
                        passage_subchunk_concat.append(subchunk)
                    else:
                        ## If the last sub-chunk is small, concat it with the next line:
                        # Old: if num_tokens(passage_subchunk_concat[-1] + token_sep + subchunk) <= normalized_chunk_length:
                        if num_tokens(passage_subchunk_concat[-1]) <= normalized_chunk_length:
                            passage_subchunk_concat[-1] += token_sep + subchunk
                        ## If the last sub-chunk is long, add it directly without trying to further subdivide.
                        else:
                            passage_subchunk_concat.append(subchunk)
                passage_chunks_split.extend(passage_subchunk_concat)
        passage_chunks = passage_chunks_split

        ## Concat small chunks into larger chunks:
        passage_chunks_concat = []
        for chunk in passage_chunks:
            ## Add the very first chunk:
            if len(passage_chunks_concat) == 0:
                passage_chunks_concat.append(chunk)
            else:
                if num_tokens(passage_chunks_concat[-1]) < normalized_chunk_length:
                    passage_chunks_concat[-1] += token_sep + chunk
                else:
                    passage_chunks_concat.append(chunk)
        passage_chunks = passage_chunks_concat
        passage_chunks_tokens = [
            cls.tokenize(passage_chunk)
            for passage_chunk in passage_chunks
        ]
        return passage_chunks, passage_chunks_tokens

    @classmethod
    def _calculate_paraphrase_score(
            cls,
            passage_tokens: List[str],
            paraphrase_tokens: List[str],
            n_gram: int
    ) -> float:
        passage_unigrams: Counter = get_ngrams(passage_tokens, n_gram=1)
        ## The number of non-contiguous overlapping tokens
        ## Example:
        ## assert lcs(
        ##     ['what', 'in', 'the', 'world', '!', 'That', '...', 'is', 'that', 'what', 'I', 'think', '...', 'this', 'is', 'amazing', '!'],
        ##     ['arguably', 'the', 'greatest', 'F1', 'driver', 'in', 'the', 'world', ',', 'Lewis', 'Hamilton', 'is', '36', 'this', 'January'],
        ## ) == (5, ['in', 'the', 'world', 'is', 'this'])
        num_tokens_overlaps, _ = lcs(passage_tokens, paraphrase_tokens)  ## Non-contiguous overlaps
        paraphrased_ngrams: Counter = get_ngrams(paraphrase_tokens, n_gram=n_gram)
        num_unique_ngrams: int = len(paraphrased_ngrams)
        paraphrased_unigrams: Counter = get_ngrams(paraphrase_tokens, n_gram=1)
        ## Can be replaced with NER step to detect unique named entities or an embedding step for different verbs:
        hallucinated_unigrams: Set[str] = {x.lower() for x in paraphrased_unigrams} \
                                          - {x.lower() for x in passage_unigrams} \
                                          - cls.STOPWORDS - cls.PUNCTUATION
        num_hallucinated_unigrams: int = len(hallucinated_unigrams)
        ## Can be replaced with NER step to detect unique named entities or an embedding step for different verbs:
        missing_unigrams: Set[str] = {x.lower() for x in passage_unigrams} \
                                     - {x.lower() for x in paraphrased_unigrams} \
                                     - cls.STOPWORDS - cls.PUNCTUATION
        num_missing_unigrams: int = len(missing_unigrams)
        paraphrase_score = round(num_unique_ngrams / np.sum([
            num_tokens_overlaps,
            num_hallucinated_unigrams,
            num_missing_unigrams
        ]), 3)
        return paraphrase_score

    @classmethod
    def _calculate_paraphrase_scores(
            cls,
            passage_tokens: List[str],
            paraphrases_list: List[str],
            paraphrase_tokens_list: List[List[str]],
            n_gram: int
    ) -> List[Tuple[float, str, List[str]]]:
        scored_paraphrases: List[Tuple[float, str, List[str]]] = []
        for paraphrase, paraphrase_tokens in zip(paraphrases_list, paraphrase_tokens_list):
            paraphrase_score: float = cls._calculate_paraphrase_score(passage_tokens, paraphrase_tokens, n_gram=n_gram)
            scored_paraphrases.append((paraphrase_score, paraphrase, paraphrase_tokens))
        return scored_paraphrases
