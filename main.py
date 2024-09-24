import argparse

import pygame

from deepqlearning import PongDeepQLearning


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Pong DQN Agent")
        parser.add_argument("--use-trained-model", action="store_true", help="Use a pre-trained model",
                            default=False)
        args = parser.parse_args()

        use_trained_model = args.use_trained_model

        dqn = PongDeepQLearning(use_trained_model=use_trained_model)
        dqn.q_learning_game_loop()

    finally:
        pygame.quit()
