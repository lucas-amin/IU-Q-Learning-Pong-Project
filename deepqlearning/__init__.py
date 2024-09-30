import random
import time

import numpy as np
import pygame
import torch
from torch import nn

from game import Game

# Define the action space
ACTION_SPACE_SIZE = 2  # Move paddle up or down


class PongDeepQLearning:
    epsilon = 50.0  # Exploration rate
    alpha = 0.01  # Learning rate (optimizer learning rate handled separately)
    gamma = 0.5  # Discount factor
    epsilon_decay = 0.999  # Decay rate for exploration
    default_path = "dqn_model.pth"
    state_size = 2
    action_space_size = 2

    def __init__(self, use_trained_model):
        self.model = self.build_model(PongDeepQLearning.state_size,
                                      PongDeepQLearning.action_space_size)

        # Define the Deep Q-Network (DQN) model using PyTorch
        if use_trained_model:
            # Load pretrained model from the default path
            self.model.load_state_dict(torch.load(self.default_path))
            self.saved = True
            self.epsilon = 0

        # Use an optimizer to update the model's weights during training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    @staticmethod
    def loss_function(state_action_values, expected_state_action_values):
        """
        Calculate mean absolute error between each element in predicted and target state action values.
        Args:
            state_action_values (tensor):
            target_state_action_values (tensor):

        Returns:
            l1_loss (float):
        """
        loss_function = nn.MSELoss()
        loss = loss_function(state_action_values, expected_state_action_values)

        return loss

    def build_model(self, state_size, action_space_size):
        model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Fully connected layer with 24 units
            nn.ReLU(),  # ReLU activation
            nn.Linear(64, 16),  # Fully connected layer with 24 units
            nn.ReLU(),  # ReLU activation
            nn.Linear(16, action_space_size)
        )
        return model

    def predict_q_values(self, state):
        # Convert state to a PyTorch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Forward pass through the DQN model to get Q-values
        q_values = self.model(state_tensor)
        return q_values.detach().numpy()[0]  # Detach and convert back to numpy array

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randrange(ACTION_SPACE_SIZE)  # Explore (random action)
        else:
            q_values = self.predict_q_values(state)
            return np.argmax(q_values)  # Exploit (action with highest Q-value)

    def update(self, state, action, reward, next_state):
        # Convert state and next_state to PyTorch tensors
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

        # Predict Q-values for current and next states
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        # Calculate the target Q-value
        with torch.no_grad():
            max_next_q_value = next_q_values.max(1)[0].item()
        target_q_value = reward + self.gamma * max_next_q_value

        # Compute the loss (difference between predicted and target Q-value)
        q_value = q_values[0][action]
        loss = self.loss_function(q_value, torch.tensor(target_q_value))  # Use L2 loss or Huber loss

        # Backpropagate the loss and update the DQN model's weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def _get_state(game):
        # Extract relevant game state information
        state = np.array([game.ball_y, game.player2_y])
        return state

    @staticmethod
    def _get_state_size(game):
        return len(PongDeepQLearning._get_state(game))

    def q_learning_game_loop(self):
        game = Game()
        saved = False

        self._get_state_size(game)

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            state = self._get_state(game)
            action = self.choose_action(state, self.epsilon)

            # Perform the action in the game environment
            game.execute_ai_action(action, player=2)

            # Update game state and get reward and next state
            game_point = game.main_loop(display=True, player1="human", player2="ai")

            # Only update the DQN model when epsilon is greater than 0.01
            if self.epsilon > 0.01:
                if abs(game.player2_y + game.paddle_height / 2 - game.ball_y) < game.paddle_height // 4:
                    reward = 20
                elif game.player2_dx > 0 and game.player2_y < game.ball_y:
                    reward = 20
                elif game.player2_dx < 0 and game.player2_y > game.ball_y:
                    reward = 20
                else:
                    reward = -10

                next_state = self._get_state(game)

                # Update the DQN only when epsilon is greater than 0.01
                self.update(state, action, reward, next_state)

                # Decay exploration rate
                self.epsilon *= self.epsilon_decay
            else:
                time.sleep(0.002)

            # Save the model
            if self.epsilon < 0.01 and not saved:
                print("Saved!")
                torch.save(self.model.state_dict(), self.default_path)
                saved = True


if __name__ == "__main__":
    try:
        pygame.init()

        game = Game()

        dqn = PongDeepQLearning(state_size=2, action_space_size=2, use_trained_model=True)
        dqn.q_learning_game_loop()

    finally:
        pygame.quit()
