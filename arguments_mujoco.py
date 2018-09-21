import argparse

def achieve_arguments():
    parse = argparse.ArgumentParser()
    # Standard Arguments
    parse.add_argument('--model-type', type=str, default='bw', help='Which model to use: "vanilla", "sil", "bw"' )
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=43, help='the random seeds')
    parse.add_argument('--env-name', type=str, default='HalfCheetah-v2', help='the environment name')
    parse.add_argument('--lr', type=float, default=5e-4, help='learning rate of the algorithm')
    parse.add_argument('--value-loss-coef', type=float, default=0.5, help='the coefficient of value loss in A2C update')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=2000000, help='the total frames(timesteps) for training')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--nsteps', type=int, default=5, help='the steps to update the network')
    parse.add_argument('--num-processes', type=int, default=16, help='the number of cpu you use')
    parse.add_argument('--entropy-coef', type=float, default=0.01, help='entropy-reg')
    parse.add_argument('--log-interval', type=int, default=1000, help='the log interval')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='the grad clip')
    parse.add_argument('--log-dir', type=str, default='logs/', help='the log dir')
    parse.add_argument('--vis', action= 'store_true', help='Plot on Visdom Server')
    parse.add_argument('--vis-interval', type=int, default=10, help='the log interval')
    
    # parse.add_argument('--eps', type=float, default=1e-5, help='param for Adam/RMS optimizer')
    # parse.add_argument('--alpha', type=float, default=0.99, help='the alpha coe of RMSprop')
    # Buffer Params for SIL and BW
    parse.add_argument('--capacity', type=int, default=100000, help='the capacity of the replay buffer')
    parse.add_argument('--sil-alpha', type=float, default=0.6, help='the exponent for PER')
    parse.add_argument('--sil-beta', type=float, default=0.1, help='sil beta')
    # SIL Specific Args
    parse.add_argument('--batch-size', type=int, default=512, help='the batch size to update the sil module')
    parse.add_argument('--n-update', type=int, default=4, help='the update of sil part')
    parse.add_argument('--mini-batch-size', type=int, default=64, help='the minimal batch size')
    parse.add_argument('--clip', type=float, default=1, help='clip parameters')
    parse.add_argument('--max-nlogp', type=float, default=5, help='max nlogp')
    parse.add_argument('--w-value', type=float, default=0.01, help='the wloss coefficient')
    # BW Specific Args
    parse.add_argument('--k-states', type=int, default=32, help='Number of top value states to train Backtracking Model on')
    parse.add_argument('--num-states', type=int, default=1, help='Number of high value state to actually backtrack on')
    parse.add_argument('--trace-size', type=int, default=5, help='Number of steps to backtrack on for a given high value state ie length of trajectory')
    parse.add_argument('--per-weight', action='store_true', help='weigh the lossed based on PER weights')
    parse.add_argument('--consistency', action='store_true', help='For consistency bw forward and backward model')
    parse.add_argument('--logclip', type=float, default=4.0, help = 'Clipping for log Normal')
    parse.add_argument('--n-a2c', type=int, default=5, help='Number of a2c updates after which to do bw update')
    parse.add_argument('--n-bw', type=int, default=20, help='Number of bw updates after which to do per n-a2c updates')
    parse.add_argument('--n-imi', type=int, default=5, help='Number of imitation updates after which to do per n-a2c updates')
    args = parse.parse_args()
    return args
