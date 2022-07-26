# Import modules
import time
import argparse
# Import custom modules
from preprocessing import preprocessing
from train import training
# Utils
from utils import str2bool, path_check, set_random_seed

def main(args):

    # Time setting
    total_start_time = time.time()

    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)

    if args.testing:
        if args.task in ['translation', 'style_transfer', 'reconstruction', 'summarization']:
            seq2seq_testing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    
    """
    ====================================original setting==========================================================
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--preprocess_path', default='./preprocessed', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/mnt/md0/dataset', type=str,
                        help='Original data path')
    parser.add_argument('--model_save_path', default='/mnt/md0/kyohoon/model_checkpoint/latent', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default='/mnt/md0/kyohoon/results/latent', type=str,
                        help='Results file path')
    # Preprocessing setting
    parser.add_argument('--tokenizer', default='alpha_map', choices=[
        'spm_unigram', 'spm_bpe', 'spm_word', 'spm_char', 'alpha_map'
            ], help='Tokenizer select; Default is alpha_map')
    parser.add_argument('--character_coverage', default=1.0, type=float,
                        help='Source language chracter coverage ratio; Default is 1.0')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=1, type=int,
                        help='Padding token index; Default is 1')
    parser.add_argument('--eos_id', default=2, type=int,
                        help='Padding token index; Default is 2')
    # Model setting
    # 0) Model selection
    parser.add_argument('--model_type', default='custom_transformer', type=str, choices=[
        'custom_transformer', 'bart', 'T5', 'bert'
            ], help='Model type selection; Default is custom_transformer')
    parser.add_argument('--isPreTrain', default=False, type=str2bool,
                        help='Using pre-trained model; Default is False')
    # 1) Custom Transformer
    parser.add_argument('--d_model', default=768, type=int, 
                        help='Transformer model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=256, type=int, 
                        help='Transformer embedding word token dimension; Default is 256')
    parser.add_argument('--n_head', default=12, type=int, 
                        help="Multihead Attention's head count; Default is 12")
    parser.add_argument('--dim_feedforward', default=2048, type=int, 
                        help="Feedforward network's dimension; Default is 2048")
    parser.add_argument('--dropout', default=0.3, type=float, 
                        help="Dropout ration; Default is 0.3")
    parser.add_argument('--embedding_dropout', default=0.1, type=float, 
                        help="Embedding dropout ration; Default is 0.1")
    parser.add_argument('--num_encoder_layer', default=8, type=int, 
                        help="Number of encoder layers; Default is 8")
    parser.add_argument('--num_decoder_layer', default=8, type=int, 
                        help="Number of decoder layers; Default is 8")
    parser.add_argument('--trg_emb_prj_weight_sharing', default=False, type=str2bool,
                        help='Weight sharing between decoder embedding and decoder linear; Default is False')
    parser.add_argument('--emb_src_trg_weight_sharing', default=False, type=str2bool,
                        help='Weight sharing between encoder embedding and decoder embedding; Default is False')
    parser.add_argument('--parallel', default=False, type=str2bool,
                        help='Transformer Encoder and Decoder parallel mode; Default is False')
    parser.add_argument('--num_common_layer', default=8, type=int, 
                        help="Number of common layers; Default is 8")
    # 2) Variational model
    parser.add_argument('--variational_mode', default=0, type=int,
                        help='Variational transformer mode; Default is False')
    parser.add_argument('--d_latent', default=128, type=int, 
                        help='Latent variable dimension; Default is 128')
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='Ralamb', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD', 'Ralamb'; Default is Ralamb")
    parser.add_argument('--scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Training setting
    parser.add_argument('--z_var', default=2, type=int)
    parser.add_argument('--min_len', default=4, type=int, 
                        help="Sentences's minimum length; Default is 4")
    parser.add_argument('--src_max_len', default=300, type=int, 
                        help="Source sentences's maximum length; Default is 300")
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help='Training epochs; Default is 100')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int,    
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    parser.add_argument('--label_smoothing_eps', default=0.05, type=float,
                        help='')
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    
    """
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--preprocess_path', default='./preprocessed', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/mnt/md0/dataset', type=str,
                        help='Original data path')
    parser.add_argument('--model_save_path', default='/mnt/md0/kyohoon/model_checkpoint/latent', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--result_path', default='/mnt/md0/kyohoon/results/latent', type=str,
                        help='Results file path')
    # Preprocessing setting
    parser.add_argument('--tokenizer', default='alpha_map', choices=[
        'spm_unigram', 'spm_bpe', 'spm_word', 'spm_char', 'alpha_map'
            ], help='Tokenizer select; Default is alpha_map')
    parser.add_argument('--character_coverage', default=1.0, type=float,
                        help='Source language chracter coverage ratio; Default is 1.0')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=1, type=int,
                        help='Padding token index; Default is 1')
    parser.add_argument('--eos_id', default=2, type=int,
                        help='Padding token index; Default is 2')
    # Model setting
    # 0) Model selection
    parser.add_argument('--model_type', default='custom_transformer', type=str, choices=[
        'custom_transformer', 'bart', 'T5', 'bert'
            ], help='Model type selection; Default is custom_transformer')
    parser.add_argument('--isPreTrain', default=False, type=str2bool,
                        help='Using pre-trained model; Default is False')
    # 1) Custom Transformer
    parser.add_argument('--d_model', default=512, type=int, 
                        help='Transformer model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=128, type=int, 
                        help='Transformer embedding word token dimension; Default is 256')
    parser.add_argument('--n_head', default=12, type=int, 
                        help="Multihead Attention's head count; Default is 12")
    parser.add_argument('--dim_feedforward', default=1024, type=int, 
                        help="Feedforward network's dimension; Default is 2048")
    parser.add_argument('--dropout', default=0.3, type=float, 
                        help="Dropout ration; Default is 0.3")
    parser.add_argument('--embedding_dropout', default=0.1, type=float, 
                        help="Embedding dropout ration; Default is 0.1")
    parser.add_argument('--num_encoder_layer', default=4, type=int, 
                        help="Number of encoder layers; Default is 8")
    parser.add_argument('--num_decoder_layer', default=8, type=int, 
                        help="Number of decoder layers; Default is 8")
    parser.add_argument('--trg_emb_prj_weight_sharing', default=False, type=str2bool,
                        help='Weight sharing between decoder embedding and decoder linear; Default is False')
    parser.add_argument('--emb_src_trg_weight_sharing', default=False, type=str2bool,
                        help='Weight sharing between encoder embedding and decoder embedding; Default is False')
    parser.add_argument('--parallel', default=False, type=str2bool,
                        help='Transformer Encoder and Decoder parallel mode; Default is False')
    parser.add_argument('--num_common_layer', default=4, type=int, 
                        help="Number of common layers; Default is 8")
    # 2) Variational model
    parser.add_argument('--variational_mode', default=0, type=int,
                        help='Variational transformer mode; Default is False')
    parser.add_argument('--d_latent', default=128, type=int, 
                        help='Latent variable dimension; Default is 128')
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='Ralamb', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD', 'Ralamb'; Default is Ralamb")
    parser.add_argument('--scheduler', default='warmup', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is warmup")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Training setting
    parser.add_argument('--z_var', default=2, type=int)
    parser.add_argument('--min_len', default=4, type=int, 
                        help="Sentences's minimum length; Default is 4")
    parser.add_argument('--src_max_len', default=300, type=int, 
                        help="Source sentences's maximum length; Default is 300")
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help='Training epochs; Default is 100')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=64, type=int,    
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    parser.add_argument('--label_smoothing_eps', default=0.05, type=float,
                        help='')
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    
    #============================additional args
    parser.add_argument('--task', default='summarization', type=str,
                        help='Model task')
    parser.add_argument('--data_name', default='bcell', type=str,
                        help='Model data_name')    

    args = parser.parse_args()

    main(args)