import argparse
import copy
import logging
import sys

from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str,
                        help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str,
                        help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--finetune_dir", default=None, help="Finetune dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str,
                        help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str,
                        help="Slot Label file")
    parser.add_argument("--domain_label_file", default="domain_label.txt", type=str,
                        help="Domain Label file")

    parser.add_argument("--no_domains", action="store_true", help="Turn off domain classification")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--device_name", default="cpu", type=str, help="cuda:0, cuda:1 or cpu")

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float,
                        help="Dropout for fully-connected layers")
    parser.add_argument("--patience", default=3, type=int, help="Max epochs without improvement")
    parser.add_argument("--ratio", default=1.0, type=float, help="Ratio of dataset for training")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--prototype", action="store_true", help="option for quick prototyping")
    parser.add_argument("--do_sieving", action="store_true", default=False,
                        help="Whether to run training data sieving.")
    parser.add_argument("--do_average_sieving", action="store_true", default=False,
                        help="Whether sieving should be based on average from slot and intent losses")
    parser.add_argument("--turnon_contrastiveloss", action="store_true", default=False,
                        help="calculate cosine similiarity, but dont add it to trainign loss")
    parser.add_argument("--dont_sieve_english", action="store_true", default=False,
                        help="dont sieve english not broken sentences. Sieve only broken english sentences")
    parser.add_argument("--do_contrastive", action="store_true", default=False,
                        help="Whether to run training data sieving.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--sieving_slot_mean", action="store_true",
                        help="use mean of slot losses in sentence")
    parser.add_argument("--eval_sieving", action="store_true",
                        help="sieving is turned off but sieving quality is evaluated for each epoch")
    parser.add_argument("--fscore_beta", type=float, help="Fscore beta factor", default=1.0)

    parser.add_argument("--hash", action="store_true",
                        help="whether to use uniq hashes from trainset")
    parser.add_argument('--sieving_coef', type=float, default=0.95,
                        help='If fscore is lower than this value, then dont do sieving')

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=2.0,
                        help='Coefficient for the slot loss.')
    parser.add_argument('--percent_broken', type=int, default=2,
                        help='how many training sentences to broke')
    parser.add_argument('--contrastive_loss_coef', type=float, default=1.0)
    parser.add_argument("--lang", default="es-ES", type=str, help="Target language")
    parser.add_argument('--sample_seed', type=int, default=128)

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignore when calculate loss)")

    parser.add_argument("--validation", default="loss", type=str,
                        help="Metric used for validation.")
    parser.add_argument("--hidden_size", default=256, type=int, help="Size for scratch model")

    parsed_args = parser.parse_args(args)

    parsed_args.model_name_or_path = MODEL_PATH_MAP[parsed_args.model_type]
    return parsed_args


def main(raw_args):
    args = parse_args(raw_args)
    init_logger(args.model_dir)
    logger = logging.getLogger(__name__)
    set_seed(args)
    tokenizer = load_tokenizer(args)

    if args.prototype:
        train_dataset, train_processor = load_and_cache_examples(args, tokenizer, mode="train", ratio=args.ratio)
        dev_dataset = train_dataset
        test_dataset = train_dataset
        test_dataset_l1 = train_dataset
        test_dataset_l2 = train_dataset
    else:
        train_dataset, train_processor = load_and_cache_examples(args, tokenizer, mode="train", ratio=args.ratio)
        dev_dataset, dev_processor = load_and_cache_examples(args, tokenizer, mode="dev")
        test_dataset, test_processor = load_and_cache_examples(args, tokenizer, mode="test")
        test_dataset_l1, testl1_processor  = load_and_cache_examples(args, tokenizer, mode="test")
        test_dataset_l2, testl2_processor = load_and_cache_examples(args, tokenizer, mode="test")


    if args.do_train:
        trainer = Trainer(
            copy.deepcopy(args),
            copy.deepcopy(train_dataset),
            copy.deepcopy(dev_dataset),
            copy.deepcopy(test_dataset),
            copy.deepcopy(test_dataset_l1),
            copy.deepcopy(test_dataset_l2),
            copy.deepcopy(tokenizer),
            copy.deepcopy(train_processor)
        )

        if args.finetune_dir:
            trainer.load_dataset()
            trainer.args.model_dir = args.finetune_dir

        set_seed(args)
        trainer.train()
        trainer.evaluate("test", per_label=False)
        trainer.evaluate("testl1", per_label=False)
        trainer.evaluate("testl2", per_label=False)

        if args.eval_sieving:
            args.eval_sieving = False
            logger.info(f"Best sieving fscore found in --eval_sieving mode: {trainer.best_sieving_fscore}")
            args.sieving_coef = trainer.best_sieving_fscore - 0.01
            logger.info(f"Starting 2nd training with --sieving_coef={args.sieving_coef}")
            trainer = Trainer(args, train_dataset, dev_dataset, test_dataset, test_dataset_l1, test_dataset_l2, tokenizer, train_processor)

            set_seed(args)
            trainer.train()

            trainer.make_plots()
            trainer.evaluate("test", per_label=False)
            trainer.evaluate("testl1", per_label=False)
            trainer.evaluate("testl2", per_label=False)
            


    
    
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("train", per_label=True)
        trainer.evaluate("dev", per_label=True)
        trainer.evaluate("test", per_label=True)

        



if __name__ == '__main__':
    main(sys.argv[1:])
