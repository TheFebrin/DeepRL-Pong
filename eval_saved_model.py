import gym
import torch
import time
import argparse
from tqdm import tqdm

from models.dqn_model import DQN
from utils.phi import preprocess, phi
from utils.utils import action_from_model_prediction, action_from_trinary_to_env


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for evaluating saved model.'
    )
    parser.add_argument(
        '--model-name', required=True, type=str, help='Name of the model.'
    )
    parser.add_argument(
        '--n-games', required=True, type=int, help='Number of games to be played.'
    )
    parser.add_argument(
        '--visualize', required=False, default=False, action='store_true', help='Show games.'
    )
    parser.add_argument(
        '--sleep', required=False, type=float, default=0.05, help='Sleep between the frames.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    n_games = args.n_games
    visualize = args.visualize
    sleep_duration = args.sleep

    print(f'Evaluating model: {model_name} on {n_games} games.')

    model = DQN(in_channels=4, out_dim=3)
    model.load_state_dict(torch.load(f'models/saved_models/{model_name}', map_location=torch.device('cpu')))

    env = gym.make("Pong-v0")
    total_score = 0
    scores_history = []
    for i in tqdm(range(n_games), desc='Validating model...'):
        observation = env.reset()
        done = False
        score = 0
        preprocessed_sequence = [preprocess(observation)]
        phi_value = phi(preprocessed_sequence)  # 84 x 84 x 4

        while not done:
            if visualize:
                time.sleep(sleep_duration)
                env.render()

            logits = model.forward_np_array(x=phi_value, device='cpu')
            action = action_from_model_prediction(x=logits)

            reward = 0
            for _ in range(4):
                observation, partial_reward, done, info = env.step(action_from_trinary_to_env(action))
                reward += partial_reward
                if done:
                    break

            score += reward

            if done:
                observation = env.reset()
                total_score += score
                scores_history.append(score)
            else:
                preprocessed_sequence.append(preprocess(observation))
                preprocessed_sequence = preprocessed_sequence[-4:]  # we need only the last 4 observations
                phi_value = phi(preprocessed_sequence)

    env.close()
    print(f'Model: {model_name} | n_games: {n_games} | Average score: {total_score / n_games :.2f}'
          f' | Min: {min(scores_history)} | Max: {max(scores_history)}')


if __name__ == '__main__':
    main()
