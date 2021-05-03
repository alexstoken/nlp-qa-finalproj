from typing import *
from utils import *
import ast, logging, tensorboard as tb, os, re, argparse, json, io, copy
from abc import ABC, abstractmethod
from collections import Counter
from pprint import pprint
from AbstractParaphraser import AbstractParaphraser
from AbstractiveSummarizationParaphraser import PegasusParaphraser
from MachineTranslationParaphraser import FairSeqParaphraser


def paraphrase(args):
    if args.verbose:
        set_root_logger()

    print(f'Paraphrasing using args:')
    pprint(vars(args))

    (meta, examples) = load_dataset(args.input_path)
    ## 'meta' example: {'header': {'dataset': 'NewsQA', 'split': 'dev'}}
    dataset_name = f"{meta['header']['dataset']}_{meta['header']['split']}"

    output_file_name = f'{dataset_name}'
    output_file_name += f'-{args.paraphrase}'
    if args.architecture == 'pegasus':
        ParaphraserSubclass: AbstractParaphraser.__class__ = PegasusParaphraser
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
    print(f'We will write to:\n  {output_path}\n  {output_args_path}')

    ## Setup device:
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.use_gpu:
        device = args.device
        assert isinstance(device, int)
        assert args.device >= 0
        torch.cuda.set_device(device)
        device = torch.device('cuda')
    print(f'We are running on {str(device)}')

    ## Load paraphraser to memory
    paraphraser: AbstractParaphraser = ParaphraserSubclass(args, device)

    paraphrased_examples = []
    for i, example in enumerate(examples):
        ## Options ['question', 'around_answer_sent', 'answer', 'answer_sent']
        if args.paraphrase == 'question':
            logging.info('=' * 50)
            logging.info('=' * 50)
            logging.info(f'Paraphrasing question#{i}...\n')
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

    ## Logging args:
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to print every paraphrase, top and bottom paraphrases, etc.',
    )

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

    ## AbstractParaphraser args
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        choices=['fairseq', 'pegasus', 'bart'],
        help='which paraphrasing architecture to use',
    )
    parser.add_argument(
        '--pretrained_model_name',
        type=str,
        choices=PegasusParaphraser.PRETRAINED_MODEL_NAMES + FairSeqParaphraser.PRETRAINED_MODEL_NAMES,
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
        type=int,
        default=3,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--print_bottom_k_paraphrases',
        type=int,
        default=1,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--paraphrase_joiner',
        type=str,
        default='\n',
        help='output dataset path. Should be a .jsonl.gz file',
    )
    paraphrase(parser.parse_args())
