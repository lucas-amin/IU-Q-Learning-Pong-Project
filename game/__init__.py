import random

import pygame


class Game:
    screen_width = 800
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pong")

    # Colors
    white = (255, 255, 255)
    black = (0, 0, 0)

    # Paddle dimensions
    paddle_width = 10
    paddle_height = 100

    # Player 1 paddle
    player1_x = 20
    player1_y = (screen_height - paddle_height) // 2
    player1_speed = 6
    player1_dx = 0

    # Player 2 paddle
    player2_x = screen_width - paddle_width - 20
    player2_y = (screen_height - paddle_height) // 2
    player2_speed = 6
    player2_dx = 0

    # Ball dimensions
    ball_size = 10
    ball_x = screen_width // 2
    ball_y = screen_height // 2
    ball_dx = 3
    ball_dy = 3

    def main_loop(self, display=False, player1="human", player2="ai"):
        # Player movement
        keys = pygame.key.get_pressed()
        self.get_action(keys, player1, player2)

        # Ball movement and collision
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        if self.ball_y <= 0 or self.ball_y >= self.screen_height - self.ball_size:
            self.ball_dy *= -1

        # Check for paddle collisions
        if (self.player1_x + self.paddle_width >= self.ball_x >= self.player1_x) and (
                self.player1_y <= self.ball_y <= self.player1_y + self.paddle_height):
            self.ball_dx *= -1
        if (self.player2_x - self.paddle_width <= self.ball_x <= self.player2_x) and (
                self.player2_y <= self.ball_y <= self.player2_y + self.paddle_height):
            self.ball_dx *= -1

        # Draw everything
        if display:
            self.display()

        # Check for scoring
        if self.ball_x <= 0:
            # Player 2 scores
            self.ball_x = self.screen_width // 2
            self.ball_y = random.randint(0, self.screen_height)
            self.ball_dx = 3
            return 1
        if self.ball_x >= self.screen_width:
            # Player 1 scores
            self.ball_x = self.screen_width // 2
            self.ball_y = random.randint(0, self.screen_height)
            self.ball_dx = 3
            return 2
        return 0

    def display(self):
        self.screen.fill(self.black)
        pygame.draw.rect(self.screen, self.white,
                         (self.player1_x, self.player1_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, self.white,
                         (self.player2_x, self.player2_y, self.paddle_width, self.paddle_height))
        pygame.draw.circle(self.screen, self.white, (self.ball_x, self.ball_y), self.ball_size)
        pygame.display.flip()

    def get_action(self, keys, player1, player2):
        if player1 == "human":
            if keys[pygame.K_w]:
                self.player1_y -= self.player1_speed
            if keys[pygame.K_s]:
                self.player1_y += self.player1_speed
        if player2 == "human":
            if keys[pygame.K_UP]:
                self.player2_y -= self.player2_speed
            if keys[pygame.K_DOWN]:
                self.player2_y += self.player2_speed

    def execute_ai_action(self, action, player=1):
        # Execute action (0: up, 1: down, 2: stay)
        if player == 2:
            if action == 0 and self.player2_y > 0:
                self.player2_y -= self.player2_speed
                self.player2_dx = - self.player1_speed
            elif action == 1 and self.player2_y < self.screen_height - self.paddle_height:
                self.player2_y += self.player2_speed
                self.player2_dx = self.player1_speed
        else:
            if action == 0 and self.player1_y > 0:
                self.player1_y -= self.player1_speed
                self.player1_dx = - self.player1_speed
            elif action == 1 and self.player1_y < self.screen_height - self.paddle_height:
                self.player1_y += self.player1_speed
                self.player1_dx = self.player1_speed


if __name__ == "__main__":
    pygame.init()

    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.main_loop(display=True, player1="human", player2="ai")

    pygame.quit()
