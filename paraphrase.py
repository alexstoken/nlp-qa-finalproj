from typing import *
from utils import *
import ast, logging, tensorboard as tb, os, re, argparse, json, io, copy
from abc import ABC, abstractmethod
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from collections import Counter
import nltk
from pprint import pprint

nltk.download('punkt')
from nltk.tokenize import word_tokenize


class Paraphraser(ABC):
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

    def __init__(self, args):
        self.args = args

    def paraphrase(
            self,
            passage: str,
            passage_tokens: List[str],
    ) -> Tuple[str, List[str]]:
        ## Driver function
        text_chunks: List[str] = self._chunk_passage(passage, normalized_chunk_length=self.args.normalized_chunk_length)
        paraphrase_chunks: List[str] = []
        paraphrase_tokens: List[str] = []
        for text_chunk in text_chunks:
            paraphrases_list, paraphrase_tokens_list = self._generate_paraphrases(
                text_chunk,
                max_length=int(round(self.args.max_len_multiplier * len(text_chunk.split(' ')))),
            )
            paraphrase_chunk, paraphrase_tokens_chunk = self._select_paraphrase(
                passage=text_chunk,
                passage_tokens=passage_tokens,
                paraphrases_list=paraphrases_list,
                paraphrase_tokens_list=paraphrase_tokens_list,
                use_scoring_function=self.args.use_scoring_function,
                score_threshold=self.args.score_threshold,
                score_n_gram=self.args.score_n_gram,
                print_top_k=self.args.print_top_k_paraphrases,
                print_bottom_k=self.args.print_bottom_k_paraphrases,
            )
            paraphrase_chunks.append(paraphrase_chunk)
            paraphrase_tokens += paraphrase_tokens_chunk
        paraphrase: str = self.args.paraphrase_joiner.join(paraphrase_chunks)
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
            question_tokens: List[str] = qa_dict['question_tokens']
            for answer_dict in qa_dict['detected_answers']:
                passage_tokens_ans_start_idx = answer_dict['token_spans'][0][0]
                passage_tokens_ans_end_idx = answer_dict['token_spans'][0][1] + 1
                ## Example `answer_tokens`:   ['Barney', 'Stinson', ',']
                answer_tokens: List[str] = passage_tokens[passage_tokens_ans_start_idx:passage_tokens_ans_end_idx]
                ## Example `answer`:   "Barney Stinson,"
                answer: str = answer_dict['text']
                passage_ans_start_idx = answer_dict['char_spans'][0][0]
                passage_ans_end_idx = answer_dict['char_spans'][0][1] + 1
                assert answer == passage[passage_ans_start_idx:passage_ans_end_idx]
                assert isinstance(answer_tokens, list)
            ## TODO: update the question text and the question tokens after paraphrasing.
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
                print()
                print('=' * 50)
                print(f'Original:\n{passage}')
            if print_top_k > 0:
                print(f'Top {print_top_k} paraphrases by scores (n_gram={score_n_gram}):')
                for paraphrase_score, paraphrase, paraphrase_tokens in scored_paraphrases[:print_top_k]:
                    print((paraphrase_score, paraphrase), end='\n\n')
            if print_bottom_k > 0:
                print(f'Bottom {print_top_k} paraphrases by scores (n_gram={score_n_gram}):')
                for paraphrase_score, paraphrase, paraphrase_tokens in scored_paraphrases[-print_bottom_k:]:
                    print((paraphrase_score, paraphrase), end='\n\n')
            paraphrase_score, paraphrase, paraphrase_tokens = scored_paraphrases[0]  ## Select those with highest score
            if paraphrase_score >= score_threshold:
                return paraphrase, paraphrase_tokens
            else:
                return None, None
        else:
            ## Return the one with the highest beam search score:
            scored_paraphrases = [
                (i + 1, paraphrase, paraphrase_tokens)  ## First element is paraphrase rank
                for i, (paraphrase, paraphrase_tokens) in enumerate(zip(paraphrases_list, paraphrase_tokens_list))
            ]
            if print_top_k > 0 or print_bottom_k > 0:
                print()
                print('=' * 50)
                print(f'Original:\n{passage}')
            if print_top_k > 0:
                print(f'Top {print_top_k} paraphrases by beam search:')
                for paraphrase_rank, paraphrase, paraphrase_tokens in scored_paraphrases[:print_top_k]:
                    print((paraphrase_rank, paraphrase), end='\n\n')
            if print_bottom_k > 0:
                print(f'Bottom {print_top_k} paraphrases by beam search:')
                for paraphrase_rank, paraphrase, paraphrase_tokens in scored_paraphrases[-print_bottom_k:]:
                    print((paraphrase_rank, paraphrase), end='\n\n')
            ## Return those with highest rank:
            return paraphrases_list[0], paraphrase_tokens_list[0]

    @staticmethod
    def _chunk_passage(
            passage: str,
            normalized_chunk_length: int,
            token_sep: str = ' ',
            line_sep: str = '.'
    ) -> List[str]:
        """
        Chunk the passage before feeding it into the Paraphraser.
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
        return passage_chunks

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
        ngram_unique_counts: int = len(paraphrased_ngrams)
        ## Can be replaced with NER step to detect unique named entities or an embedding step for different verbs:
        hallucinated_unigrams: Set[str] = set(paraphrased_ngrams.keys()) \
                                          - set(passage_unigrams.keys()) \
                                          - cls.STOPWORDS - cls.PUNCTUATION
        paraphrase_score = round(ngram_unique_counts / (num_tokens_overlaps + hallucinated_unigrams), 3)
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


class AbstractiveSummarizationParaphraser(Paraphraser):
    pass


class MachineTranslationParaphraser(Paraphraser):
    pass


class PegasusParaphraser(AbstractiveSummarizationParaphraser):
    GENERATED_NEWLINE = '<n>'

    ## Found on HuggingFace search bar, from the PEGASUS paper released by Google: https://arxiv.org/pdf/1912.08777.pdf
    PRETRAINED_MODEL_NAMES = [
        'google/pegasus-large',
        'google/pegasus-xsum',
        'google/pegasus-cnn_dailymail',
        'google/pegasus-multi_news',
    ]

    def __init__(self, args):
        super().__init__(args)
        assert args.pretrained_model_name in self.PRETRAINED_MODEL_NAMES
        self.pretrained_model_name = args.pretrained_model_name
        model = PegasusForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        if args.use_gpu:
            model = cuda(args, model)
        self.model: PegasusForConditionalGeneration = model
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
        )
        tokenized_ids = cuda(self.args, tokenized_ids)
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
            word_tokenize(paraphrase) for paraphrase in paraphrases_list
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


def paraphrase(args):
    print(f'Paraphrasing using args:')
    pprint(vars(args))

    (meta, examples) = load_dataset(args.input_path)
    ## 'meta' example: {'header': {'dataset': 'NewsQA', 'split': 'dev'}}
    dataset_name = f"{meta['header']['dataset']}_{meta['header']['split']}"

    output_file_name = f'{dataset_name}'
    output_file_name += f'-{args.paraphrase}'
    if args.architecture == 'pegasus':
        paraphraser: Paraphraser = PegasusParaphraser(args)
        output_file_name += f'-{args.pretrained_model_name.replace("google/", "")}'
        output_file_name += f'-chk{args.normalized_chunk_length}'
        output_file_name += f'-mul{args.max_len_multiplier}'
        output_file_name += f'-beam{args.num_beams}'
        output_file_name += f'-temp{args.temperature}'
    else:
        raise NotImplementedError(f'Unsupported architecture: {args.architecture}')

    if args.use_scoring_function:
        output_file_name += f'-score'
        output_file_name += f'-th{args.score_threshold}'
        output_file_name += f'-ngram{args.score_n_gram}'

    output_dir = os.sep.join(args.input_path.split(os.sep)[:-1])
    output_path = os.path.join(output_dir, output_file_name, '.jsonl.gz')
    output_args_path = os.path.join(output_dir, output_file_name, '.args.json')

    paraphrased_examples = []
    for example in examples:
        ## Options ['question', 'around_answer_sent', 'answer', 'answer_sent']
        if args.paraphrase == 'question':
            paraphrased_examples.append(paraphraser.paraphrase_question(example))
        elif args.paraphrase == 'around_answer_sent':
            paraphrased_examples.append(paraphraser.paraphrase_around_answer_sent(example))
        elif args.paraphrase == 'answer':
            paraphrased_examples.append(paraphraser.paraphrase_question(example))
        elif args.paraphrase == 'answer_sent':
            paraphrased_examples.append(paraphraser.paraphrase_question(example))
        else:
            raise NotImplementedError(f'Cannot paraphrase "{args.paraphrase}"')

    with gzip.open(output_path, 'wb') as out:
        for ex in paraphrased_examples:
            ## Ref: https://stackoverflow.com/a/39451012
            out.write((json.dumps(ex) + '\n').encode('utf-8'))

    with io.open(output_args_path, 'w') as out:
        out.write(json.dumps(vars(args)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Training args.
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='whether to use GPU',
    )

    ## Data args:
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='input dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--paraphrase',
        type=str,
        required=True,
        choices=['question', 'around_answer_sent', 'answer', 'answer_sent'],
        help='what to paraphrase',
    )

    ## Paraphraser args
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        choices=['backtranslate', 'pegasus', 'bart'],
        help='which paraphrasing architecture to use',
    )
    parser.add_argument(
        '--pretrained_model_name',
        type=str,
        choices=PegasusParaphraser.PRETRAINED_MODEL_NAMES,
        help='which pretrained model to use',
    )

    parser.add_argument(
        '--normalized_chunk_length',
        type=int,
        default=20,
        help='Approx. length (in words) each input to the paraphraser should be.',
    )
    parser.add_argument(
        '--max_len_multiplier',
        type=float,
        default=1.5,
        help='The max allowed length of the paraphrased output, relative to the input length',
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=75,
        help='Number of beams for beam search',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.5,
        help='Temperature ',
    )
    parser.add_argument(
        '--use_scoring_function',
        type=bool,
        default=True,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=1.2,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--score_n_gram',
        type=float,
        default=1,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--print_top_k_paraphrases',
        type=bool,
        default=False,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--print_bottom_k_paraphrases',
        type=bool,
        default=False,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--paraphrase_joiner',
        type=str,
        default='\n',
        help='output dataset path. Should be a .jsonl.gz file',
    )
    paraphrase(parser.parse_args())
