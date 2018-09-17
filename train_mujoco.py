from baselines.common.cmd_util import make_vec_env, make_mujoco_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from arguments_mujoco import achieve_arguments
from a2c_agent_mujoco import a2c_agent
from baselines import logger

if __name__ == '__main__':
    args = achieve_arguments()
    #Logger helps to set correct logs for using tensorboard later
    # logger.configure(dir=args.log_dir)
    # create environments
    env_args = {'episode_life': False, 'clip_rewards': False}
    #VecFrameStacks is Frame-Stacking with 4 frames for atari environments
    #make_vec_env will make and wrap atari environments correctly in a vectorized form. Here we are also doing multiprocessing training with multiple environments
    envs = make_vec_env(args.env_name, 'mujoco', args.num_processes, args.seed, wrapper_kwargs=env_args)
    trainer = a2c_agent(envs, args)
    trainer.learn()
    envs.close()
