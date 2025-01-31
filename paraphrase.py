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
    elif args.architecture == 'fairseq':
        ParaphraserSubclass: AbstractParaphraser.__class__ = FairSeqParaphraser
        output_file_name += f'-fairseq'
        output_file_name += f'-chk{args.normalized_chunk_length}'
        output_file_name += f'-mul{args.max_len_multiplier}'
        output_file_name += f'-beam{args.num_beams}'
        output_file_name += f'-temp{args.temperature}'  ## We should have added this earlier.
        if args.MT_sampling:
            output_file_name += f'-MT_sampling_k{args.MT_sampling_topk}'
    else:
        raise NotImplementedError(f'Unsupported architecture: {args.architecture}')

    if args.use_scoring_function:
        output_file_name += f'-score'
        output_file_name += f'-th{args.score_threshold}'
        output_file_name += f'-ngram{args.score_n_gram}'

    output_dir = os.sep.join(args.input_path.split(os.sep)[:-1])

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

    if isinstance(args.examples_range, str) and args.examples_range.strip().lower() != 'start:end':
        ## Example ranges:
        ## 1:100
        ## 101:200
        ## 201:300
        ## etc
        range_start_idx = args.examples_range.split(':')[0].strip()
        if range_start_idx == '' or range_start_idx.lower() == 'start':
            range_start_idx = 0
        range_start_idx = max(0, int(range_start_idx) - 1)  ## range_start_idx must be a valid index
        range_end_idx = args.examples_range.split(':')[1].strip()
        if range_end_idx == '' or range_end_idx.lower() == 'end':
            range_end_idx = len(examples)
        range_end_idx = min(len(examples) - 1, int(range_end_idx) - 1)  ## range_end_idx must be a valid index
        examples = examples[range_start_idx:range_end_idx + 1]
        print(f'We will only paraphrase from {range_start_idx + 1} to {range_end_idx + 1} ({len(examples)} examples)')
        output_file_name += f'-range_{range_start_idx + 1}_{range_end_idx + 1}'
        meta['header']['examples_range'] = f'{range_start_idx + 1}:{range_end_idx + 1}'

    print('~'*100)
    print('~'*100)
    print(f'{output_file_name}')
    print('~'*100)
    print('~'*100)

    paraphrased_examples = []
    start = time.time()
    batch_size = 1  ## TODO: implement bigger batching for faser paraphrasing.
    for batch_idx, batch_start in enumerate(range(0, len(examples), batch_size)):
        batched_examples = examples[batch_start:batch_start + batch_size]
        if batch_size == 1:
            batched_examples = batched_examples[0]

        logging.info('=' * 50)
        logging.info('=' * 50)
        logging.info(f'Paraphrasing example {batch_start + 1} of {len(examples)}...\n')

        if args.paraphrase == 'question':
            paraphrased_examples.append(paraphraser.paraphrase_question(batched_examples))
        elif args.paraphrase == 'around_answer_sent':
            paraphrased_examples.append(paraphraser.paraphrase_around_answer_sent(batched_examples))
        elif args.paraphrase == 'answer':
            paraphrased_examples.append(paraphraser.paraphrase_answer(batched_examples))
        elif args.paraphrase == 'answer_sent':
            paraphrased_examples.append(paraphraser.paraphrase_answer_sent(batched_examples))
        else:
            raise NotImplementedError(f'Cannot paraphrase "{args.paraphrase}"')
        if (batch_idx + 1) % 10 == 0:
            now = time.time()
            print(f'\n{output_file_name}: Took {(now - start):.3f} seconds total to '
                  f'paraphrase {batch_start + 1} of {len(examples)} examples '
                  f'({(now - start) / (batch_start + batch_size + 1):.3f} seconds/example).')
        if args.num_checkpoints != 0 and (batch_idx + 1) % (len(examples) // args.num_checkpoints) == 0:
            print(f'\n{output_file_name}: Saving checkpoint with {len(paraphrased_examples)} examples.')
            with gzip.open(
                    os.path.join(output_dir, output_file_name + f'-{batch_start + batch_size + 1}' + '.jsonl.gz'),
                    'wb') as out:
                for ex in [meta] + paraphrased_examples:
                    ## Ref: https://stackoverflow.com/a/39451012
                    out.write((json.dumps(ex) + '\n').encode('utf-8'))

    output_path = os.path.join(output_dir, output_file_name + '.jsonl.gz')
    output_args_path = os.path.join(output_dir, output_file_name + '.args.json')
    print(f'We will write to:\n  {output_path}\n  {output_args_path}')

    with gzip.open(output_path, 'wb') as out:
        for ex in [meta] + paraphrased_examples:
            ## Ref: https://stackoverflow.com/a/39451012
            out.write((json.dumps(ex) + '\n').encode('utf-8'))

    with io.open(output_args_path, 'w+') as out:
        out.write(json.dumps(vars(args), indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Logging args:
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to print every paraphrase, top and bottom paraphrases, etc.',
    )
    parser.add_argument(
        '--num_checkpoints',
        type=int,
        default=20,
        help='Number of checkpoints to save while paraphrasing',
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

    ## This arg is ignored, 1 is always used:
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='batch size to pass to paraphrasers',
    )

    parser.add_argument(
        '--examples_range',
        type=str,
        default='start:end',
        help='Range of examples to process. By default, does everything',
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
        default=False,
        action='store_true',
        help='Whether or not to use the scoring function',
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=1.2,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--score_n_gram',
        type=int,
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
        default=3,
        help='output dataset path. Should be a .jsonl.gz file',
    )
    parser.add_argument(
        '--paraphrase_joiner',
        type=str,
        default='\n',
        help='output dataset path. Should be a .jsonl.gz file',
    )

    parser.add_argument(
        '--MT_sampling',
        action='store_true',
        help='Use sampling in machine translation. If false, uses greedy beam search',
    )

    parser.add_argument(
        '--MT_sampling_topk',
        type=int,
        default=50,
        help='k for top k sampling of next work in MT. Only used when MT_sampling is True',
    )
    paraphrase(parser.parse_args())
