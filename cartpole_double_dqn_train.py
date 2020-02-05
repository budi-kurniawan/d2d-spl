from util.dqn_util import DoubleDQNAgent
import cartpole_dqn_train as base
from cartpole_dqn_train import main
from cartpole_utils import create_cartpole_env

base.DQNAgent = DoubleDQNAgent

if __name__ == '__main__':
    base.NUM_EPISODES = 1000
    base.hidden_dim = 12
    env = create_cartpole_env()
    input_dim, output_dim = base.get_env_dim(env)
    results_path = 'results/double_dqn_results'
    main(env, input_dim, output_dim, results_path)