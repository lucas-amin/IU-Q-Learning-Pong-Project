# IU-Q-Learning-Pong-Project


A Pong Game Implemented with Q-Learning

This project demonstrates a simple Pong game where one player is controlled by a human using the W and S keys, and the other player is controlled by a Q-Learning algorithm.

Installation
Clone the Repository:

```
Bash
git clone https://github.com/lucas-amin/IU-Q-Learning-Pong-Project.git
```

### Install Dependencies:

Bash
```
cd IU-Q-Learning-Pong-Project
pip install -r requirements.txt
```

Usage
Run the Game:
```
Bash
python main.py
```

### Technologies Used

* PyTorch: Deep learning framework for the Q-Learning agent.
* Pygame: Game development library for the Pong game environment.
* NumPy: Numerical computing library for array operations.

### Project Structure

* main.py: Main script to run the game and train the agent.
* pong.py: Defines the Pong game environment.
* agent.py: Implements the Q-Learning agent.
* requirements.txt: List of required Python packages.

### Game Controls
* Human Player: W and S keys to control the paddle.
* AI Player: Controlled by the Q-Learning algorithm.

### Future Improvements
* Optimize Q-Learning algorithm: Explore different hyperparameters and techniques to improve agent performance.
* Implement double Q-learning: Reduce overestimation bias in Q-value estimates.
