import argparse

parser = argparse.ArgumentParser(description='ARAE for SNLI')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--kenlm_path', type=str, default='./kenlm',
                    help='path to kenlm directory')
parser.add_argument('--save', type=str, default='snli_example',
                    help='output directory name')

# Data Processing Arguments
parser.add_argument('--maxlen', type=int, default=15,
                    help='maximum length')
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                    help='not lowercase all text')
parser.set_defaults(lowercase=True)

# Model Arguments
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.05,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.9995,
                    help='anneal noise_r exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--z_size', type=int, default=100,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=12,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=5,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_ae', type=int, default=1,
                    help='number of gan-into-ae iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=1e-04,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-04,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')
parser.add_argument('--gan_gp_lambda', type=float, default=1,
                    help='WGAN GP penalty lambda')
parser.add_argument('--grad_lambda', type=float, default=0.1,
                    help='WGAN into AE lambda')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
print(vars(args))


exec(open("train.py").read())
