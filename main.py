"""Script for training and evaluating QA models.

Example command to train the (medium-sized) baseline model on SQuAD
with a GPU, and write its predictions to an output file:

Usage:
    python3 main.py \
        --use_gpu \
        --model "baseline" \
        --model_path "squad_model.pt" \
        --train_path "datasets/squad_train.jsonl.gz" \
        --dev_path "datasets/squad_dev.jsonl.gz" \
        --output_path "squad_predictions.txt" \
        --hidden_dim 256 \
        --bidirectional \
        --do_train \
        --do_test

Author:
    Shrey Desai and Yasumasa Onoe
"""

import argparse
import io
import os
import pprint
import json

import torch
import numpy as np
import pandas as pd
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data import *

from model import BaselineReader
from utils import *
from evaluate import *

_TQDM_BAR_SIZE = 75
_TQDM_LEAVE = False
_TQDM_UNIT = ' batches'
_TQDM_OPTIONS = {
    'ncols': _TQDM_BAR_SIZE, 'leave': _TQDM_LEAVE, 'unit': _TQDM_UNIT
}

parser = argparse.ArgumentParser()

# Training arguments.
parser.add_argument('--device', type=int, default=0)
parser.add_argument(
    '--use_gpu',
    action='store_true',
    help='whether to use GPU',
)
parser.add_argument(
    '--model',
    type=str,
    required=True,
    choices=['baseline'],
    help='which model to use',
)
parser.add_argument(
    '--model_path',
    type=str,
    default=f'models/{generate_random_string()}.pt',
    help='path to load/save model checkpoints',
)
parser.add_argument(
    '--embedding_path',
    type=str,
    default='glove/glove.6B.300d.txt',
    help='GloVe embedding path',
)
parser.add_argument(
    '--train_path',
    type=str,
    nargs='+',
    required=True,
    help='training dataset path or paths',
)
parser.add_argument(
    '--dev_path',
    type=str,
    required=True,
    help='dev dataset path',
)
parser.add_argument(
    '--num_answers',
    type=int,
    default=1,
    help='Max number of answers per question',
)
parser.add_argument(
    '--lowercase_passage',
    type=bool,
    default=True,
    help='whether to lowercase the passage text',
)
parser.add_argument(
    '--lowercase_question',
    type=str,
    default=True,
    help='whether to lowercase the question text',
)
parser.add_argument(
    '--max_context_length',
    type=int,
    default=1024,
    help='maximum context length (do not change!)',
)
parser.add_argument(
    '--max_question_length',
    type=int,
    default=64,
    help='maximum question length (do not change!)',
)
parser.add_argument(
    '--output_path',
    type=str,
    default=f'preds/{generate_random_string()}.preds.txt',
    help='predictions output path',
)
parser.add_argument(
    '--runs',
    type=int,
    default=3,
    help='Number of runs',
)
parser.add_argument(
    '--shuffle_examples',
    action='store_true',
    help='shuffle training example at the beginning of each epoch',
)
parser.add_argument('--skip_no_answer', action='store_true')

parser.add_argument(
    '--data_aug',
    type=str,
    choices=['paraphrase', 'backtranslate', 'both'],
    help='which data augmentation scheme to use in training')

parser.add_argument(
    '--paraphrase_rate',
    type=float,
    default=1.0,
    help='Rate at which to sample paraphrases. Default is to always use paraphrase.',
)

parser.add_argument(
    '--paraphrase_score_thresh',
    type=float,
    default=0.0,
    help='Threshold to consider paraphrase good. Defaults to accepting all paraphrases. Good value is 1.2.',
)

parser.add_argument(
    '--paraphrase_score_n_gram',
    type=int,
    default=1,
    help='n_gram for scoring fn',
)

# Optimization arguments.
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='number of training epochs',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='training and evaluation batch size',
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help='training learning rate',
)
parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.,
    help='training weight decay',
)
parser.add_argument(
    '--grad_clip',
    type=float,
    default=0.5,
    help='gradient norm clipping value',
)
parser.add_argument(
    '--early_stop',
    type=int,
    default=3,
    help='number of epochs to wait until early stopping',
)
parser.add_argument(
    '--do_train',
    action='store_true',
    help='flag to enable training',
)
parser.add_argument(
    '--do_test',
    action='store_true',
    help='flag to enable testing',
)
parser.add_argument(
    '--do_train_test_eval',
    action='store_true',
    help='flag to enable training, test and eval',
)

# Model arguments.
parser.add_argument(
    '--vocab_size',
    type=int,
    default=50000,
    help='vocabulary size (dynamically set, do not change!)',
)
parser.add_argument(
    '--embedding_dim',
    type=int,
    default=300,
    help='embedding dimension',
)
parser.add_argument(
    '--hidden_dim',
    type=int,
    default=256,
    help='hidden state dimension',
)
parser.add_argument(
    '--rnn_cell_type',
    choices=['lstm', 'gru'],
    default='lstm',
    help='Type of RNN cell',
)
parser.add_argument(
    '--bidirectional',
    action='store_true',
    help='use bidirectional RNN',
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.,
    help='dropout on passage and question vectors',
)


def _print_arguments(args):
    """Pretty prints command line args to stdout.

    Args:
        args: `argparse` object.
    """

    args_dict = vars(args)
    pprint.pprint(args_dict)


def _select_model(args):
    """
    Selects and initializes model. To integrate custom models, (1)
    add the model name to the parser choices above, and (2) modify
    the conditional statements to include an instance of the model.

    Args:
        args: `argparse` object.

    Returns:
        Instance of a PyTorch model supplied with args.
    """
    if args.model == 'baseline':
        return BaselineReader(args)
    else:
        raise RuntimeError(f'model \'{args.model}\' not recognized!')


def _early_stop(args, eval_history):
    """
    Determines early stopping conditions. If the evaluation loss has
    not improved after `args.early_stop` epoch(s), then training
    is ended prematurely. 

    Args:
        args: `argparse` object.
        eval_history: List of booleans that indicate whether an epoch resulted
            in a model checkpoint, or in other words, if the evaluation loss
            was lower than previous losses.

    Returns:
        Boolean indicating whether training should stop.
    """
    return (len(eval_history) > args.early_stop and not any(eval_history[-args.early_stop:]))


def _calculate_loss(start_logits, end_logits, start_positions, end_positions):
    """
    Calculates cross-entropy loss for QA samples, which is defined as
    the mean of the loss values incurred by the starting and ending position
    distributions when compared to the gold endpoints.

    Args:
        start_logits: Predicted distribution over start positions.
        end_logits: Predicted distribution over end positions.
        start_positions: Gold start positions.
        end_positions: Gold end positions.

    Returns:
        Loss value for a batch of sasmples.
    """
    # If the gold span is outside the scope of the maximum
    # context length, then ignore these indices when computing the loss.
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)

    # Compute the cross-entropy loss for the start and end logits.
    criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = criterion(start_logits, start_positions)
    end_loss = criterion(end_logits, end_positions)

    return (start_loss + end_loss) / 2.


def train(args, epoch, model, dataset):
    """
    Trains the model for a single epoch using the training dataset.

    Args:
        args: `argparse` object.
        epoch: Epoch number (used in the `tqdm` bar).
        model: Instance of the PyTorch model.
        dataset: Training dataset.

    Returns:
        Training cross-entropy loss normalized across all samples.
    """
    # Set the model in "train" mode.
    model.train()

    # Cumulative loss and steps.
    train_loss = 0.
    train_steps = 0

    # Set up optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Set up training dataloader. Creates `args.batch_size`-sized
    # batches from available samples.
    train_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=args.shuffle_examples),
        **_TQDM_OPTIONS,
    )

    for batch in train_dataloader:
        # Zero gradients.
        optimizer.zero_grad()

        # Forward inputs, calculate loss, optimize model.
        start_logits, end_logits = model(batch)
        loss = _calculate_loss(
            start_logits,
            end_logits,
            batch['start_positions'],
            batch['end_positions'],
        )
        loss.backward()
        if args.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Update tqdm bar.
        train_loss += loss.item()
        train_steps += 1
        train_dataloader.set_description(
            f'[train] epoch = {epoch}, loss = {train_loss / train_steps:.6f}'
        )

    return train_loss / train_steps


def evaluate_loss(args, epoch, model, dataset):
    """
    Evaluates the model for a single epoch using the development dataset.

    Args:
        args: `argparse` object.
        epoch: Epoch number (used in the `tqdm` bar).
        model: Instance of the PyTorch model.
        dataset: Development dataset.

    Returns:
        Evaluation cross-entropy loss normalized across all samples.
    """
    # Set the model in "evaluation" mode.
    model.eval()

    # Cumulative loss and steps.
    eval_loss = 0.
    eval_steps = 0

    # Set up evaluation dataloader. Creates `args.batch_size`-sized
    # batches from available samples. Does not shuffle.
    eval_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    with torch.no_grad():
        for batch in eval_dataloader:
            # Forward inputs, calculate loss.
            start_logits, end_logits = model(batch)
            loss = _calculate_loss(
                start_logits,
                end_logits,
                batch['start_positions'],
                batch['end_positions'],
            )

            # Update tqdm bar.
            eval_loss += loss.item()
            eval_steps += 1
            eval_dataloader.set_description(
                f'[eval] epoch = {epoch}, loss = {eval_loss / eval_steps:.6f}'
            )

    return eval_loss / eval_steps


def write_predictions(args, model, dataset):
    """
    Writes model predictions to an output file. The official QA metrics (EM/F1)
    can be computed using `evaluation.py`. 

    Args:
        args: `argparse` object.
        model: Instance of the PyTorch model.
        dataset: Test dataset (technically, the development dataset since the
            official test datasets are blind and hosted by official servers).
    """
    # Load model checkpoint.
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    # Set up test dataloader.
    test_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    # Output predictions.
    outputs = []

    with torch.no_grad():
        for (i, batch) in enumerate(test_dataloader):
            # Forward inputs.
            start_logits, end_logits = model(batch)

            # Form distributions over start and end positions.
            batch_start_probs = F.softmax(start_logits, 1)
            batch_end_probs = F.softmax(end_logits, 1)

            for j in range(start_logits.size(0)):
                # Find question index and passage.
                sample_index = args.batch_size * i + j
                qid = dataset.samples[sample_index][QID_KEY]
                passage = dataset.samples[sample_index][PASSAGE_TOKENS_KEY]

                # Unpack start and end probabilities. Find the constrained
                # (start, end) pair that has the highest joint probability.
                start_probs = unpack(batch_start_probs[j])
                end_probs = unpack(batch_end_probs[j])
                start_index, end_index = search_span_endpoints(
                    start_probs, end_probs
                )

                # Grab predicted span.
                pred_span = ' '.join(passage[start_index:(end_index + 1)])

                # Add prediction to outputs.
                outputs.append({'qid': qid, 'answer': pred_span})

    # Write predictions to output file.
    with open(args.output_path, 'w+') as f:
        for elem in outputs:
            f.write(f'{json.dumps(elem)}\n')


def determine_domain(paths):
    if type(paths) == list:
        path = paths[0]

    if 'news' in path.lower():
        return 'news'
    elif 'bio' in path.lower():
        return 'bio'


def base_sample_multiplier(domain, train_dataset):
    if domain == 'news':
        _, base_samples = load_dataset('datasets/newsqa_train.jsonl.gz')
    elif domain == 'bio':
        _, base_samples = load_dataset('datasets/bioasq_train.jsonl.gz')
    else:
        return None
    return len(train_dataset.samples) / len(base_samples), len(base_samples)


def calc_equivalent_epochs(epochs, batch_size, base_samples, train_samples):
    # calculate naive # of grad updates
    grad_updates = epochs * (base_samples // batch_size)
    equivalent_epochs = grad_updates / (train_samples // batch_size)
    return int(np.round(equivalent_epochs))


def main(args):
    """
    Main function for training, evaluating, and checkpointing.

    Args:
        args: `argparse` object.
    """
    # Print arguments.
    print('\nusing arguments:')
    _print_arguments(args)
    print()

    # Check if GPU is available.
    if not args.use_gpu and torch.cuda.is_available():
        print('warning: GPU is available but args.use_gpu = False')
        print()

    # Set up datasets.
    print('Creating train dataset: ')
    train_dataset = QADataset(args, args.train_path)
    print('Creating dev dataset: ')
    dev_dataset = QADataset(args, args.dev_path)

    # Create vocabulary and tokenizer.
    vocabulary = Vocabulary(train_dataset.samples, args.vocab_size)
    tokenizer = Tokenizer(vocabulary)
    for dataset in (train_dataset, dev_dataset):
        dataset.register_tokenizer(tokenizer)
    args.vocab_size = len(vocabulary)
    args.pad_token_id = tokenizer.pad_token_id
    print(f'vocab words = {len(vocabulary)}')

    if type(args.train_path) == list:
        metrics_path = 'results/'
        for idx, path in enumerate(args.train_path):
            if idx != 0:
                metrics_path += '+'
            metrics_path += path[path.find('datasets/') + len('datasets/'):path.find('.jsonl.gz')]
        metrics_path += '.metrics.txt'
    else:
        metrics_path = args.train_path.replace('.jsonl.gz', '.metrics.txt').replace('datasets', 'results')
    print(metrics_path)
    if '-score-' not in metrics_path and args.paraphrase_rate > 0.0 and args.paraphrase_score_thresh > 0:
        metric_path_suffix = '-score'
        metric_path_suffix += f'-th{args.paraphrase_score_thresh}'
        metric_path_suffix += f'-ngram{args.paraphrase_score_n_gram}'
        if args.paraphrase_rate < 1.0:
            metric_path_suffix += f'-rate{args.paraphrase_rate}'
        metrics_path = metrics_path.replace('.metrics.txt', f'{metric_path_suffix}.metrics.txt')
    else:
        metrics_path = re.sub('-score-.*', '', metrics_path)
        metric_path_suffix = '-score'
        metric_path_suffix += f'-th{args.paraphrase_score_thresh}'
        metric_path_suffix += f'-ngram{args.paraphrase_score_n_gram}'
        if args.paraphrase_rate < 1.0:
            metric_path_suffix += f'-rate{args.paraphrase_rate}'
        metrics_path += metric_path_suffix
        metrics_path += '.metrics.txt'

    os.system(f'rm {metrics_path}')
    print(f'\nWe will save results to {metrics_path}')

    # Print number of samples.
    print(f'train samples = {len(train_dataset)}')
    print(f'Num paraphrased questions in train dataset: {train_dataset.num_paraphrased_questions} '
          f'({100 * train_dataset.num_paraphrased_questions / len(train_dataset):.1f}%)')
    print('~' * 50)
    print(f'dev samples = {len(dev_dataset)}')
    print(f'Num paraphrased questions in dev dataset: {dev_dataset.num_paraphrased_questions} '
          f'({100 * dev_dataset.num_paraphrased_questions / len(dev_dataset):.1f}%)')

    metrics_list = []

    # domain = determine_domain(args.train_path)
    # dataset_multiplier, num_base_samples = base_sample_multiplier(domain, train_dataset)
    # equivalent_epochs = calc_equivalent_epochs(args.epochs, args.batch_size, num_base_samples,
    #                                            len(train_dataset.samples))
    equivalent_epochs = args.epochs
    # print(
    #     f'This dataset is {dataset_multiplier} times as large as the base training set for {domain}. Instead of {args.epochs} epochs, each run will have {equivalent_epochs} epochs.')

    for run_i in range(args.runs):
        print('=' * 100)
        print(' ' * 45 + f'RUN#{run_i + 1}')
        print('=' * 100)

        # Select model.
        model = _select_model(args)
        num_pretrained = model.load_pretrained_embeddings(
            vocabulary, args.embedding_path
        )
        pct_pretrained = round(num_pretrained / len(vocabulary) * 100., 2)
        print(f'using pre-trained embeddings from \'{args.embedding_path}\'')
        print(
            f'initialized {num_pretrained}/{len(vocabulary)} '
            f'embeddings ({pct_pretrained}%)'
        )
        print()

        if args.use_gpu:
            model = cuda(args, model)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'using model \'{args.model}\' ({params} params)')
        print(model)
        print()
        if args.do_train or args.do_train_test_eval:
            # Track training statistics for checkpointing.
            eval_history = []
            best_eval_loss = float('inf')

            # Begin training.

            # adjust epoch # by the amount of paraphrased questions concatenated
            # since we want apples to apples comparision by # of gradient updates
            for epoch in range(1, equivalent_epochs + 1):
                # Perform training and evaluation steps.
                train_loss = train(args, epoch, model, train_dataset)
                eval_loss = evaluate_loss(args, epoch, model, dev_dataset)

                # If the model's evaluation loss yields a global improvement,
                # checkpoint the model.
                eval_history.append(eval_loss < best_eval_loss)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(model.state_dict(), args.model_path)

                print(
                    f'epoch = {epoch} | '
                    f'train loss = {train_loss:.6f} | '
                    f'eval loss = {eval_loss:.6f} | '
                    f"{'saving model!' if eval_history[-1] else ''}"
                )

                # If early stopping conditions are met, stop training.
                if _early_stop(args, eval_history):
                    suffix = 's' if args.early_stop > 1 else ''
                    print(
                        f'no improvement after {args.early_stop} epoch{suffix}. '
                        'early stopping...'
                    )
                    print()
                    break

        if args.do_test or args.do_train_test_eval:
            # Write predictions to the output file. Use the printed command
            # below to obtain official EM/F1 metrics.
            write_predictions(args, model, dev_dataset)
            eval_cmd = (
                'python3 evaluate.py '
                f'--dataset_path {args.dev_path} '
                f'--output_path {args.output_path}'
            )
            print()
            print(f'predictions written to \'{args.output_path}\'')
            print(f'compute EM/F1 with: \'{eval_cmd}\'')
            print()

        if args.do_train_test_eval:
            answers = read_answers(args.dev_path)
            predictions = read_predictions(args.output_path)
            metrics = evaluate_metrics(answers, predictions, args.skip_no_answer)
            metrics['train_dataset.size'] = len(train_dataset)
            metrics['train_dataset.num_paraphrased_questions'] = train_dataset.num_paraphrased_questions
            metrics['dev_dataset.size'] = len(dev_dataset)
            metrics['dev_dataset.num_paraphrased_questions'] = dev_dataset.num_paraphrased_questions
            print(metrics)
            with io.open(metrics_path, 'a+') as out:
                out.write(json.dumps(metrics))
                out.write('\n')
            metrics_list.append(metrics)

        del model

    metrics_df = pd.DataFrame(metrics_list)
    with io.open(metrics_path, 'a+') as out:
        out.write('\nAVERAGE:\n')
        out.write(json.dumps(metrics_df.mean(axis=0).round(2).to_dict()))
        out.write('\nMIN:\n')
        out.write(json.dumps(metrics_df.min(axis=0).round(2).to_dict()))
        out.write('\nMAX:\n')
        out.write(json.dumps(metrics_df.max(axis=0).round(2).to_dict()))
    print('~' * 50)
    print('~' * 50)


if __name__ == '__main__':
    main(parser.parse_args())
