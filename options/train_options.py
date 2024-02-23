from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=160, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=200, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')

        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_false', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--load_weight', type=str, default=None,
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--num_graph_update', type=int, default=10,
                            help='the time of updating hyper-graph in each epoch. Default: update the parameters of hyper-graph 10 times in each epoch.')
        parser.add_argument('--weight_u', type=float, default=1.,
                            help='the loss weight for unlabeled data. Default: 1.0')
        parser.add_argument('--weight_center', type=float, default=0.,
                            help='the loss weight for center loss. Default: 0.')
        parser.add_argument('--train_encoders', action='store_true', help='whether fix or train the encoders.')
        parser.add_argument('--use_strong_aug', action='store_true', help='use strong augmentation to images or not.')
        self.isTrain = True
        return parser


