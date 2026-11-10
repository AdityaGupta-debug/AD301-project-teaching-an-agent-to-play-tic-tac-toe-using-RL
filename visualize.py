"""Pygame visualization for the trained Tic-Tac-Toe agent."""
import argparse
import random
from pathlib import Path

import pygame

from agent import QLearningAgent
from environment import TicTacToe

WIDTH, HEIGHT = 600, 700
BOARD_TOP, CELL = 90, 180
BG = (24, 27, 35)
GRID = (220, 224, 232)
X_COLOR = (92, 178, 255)
O_COLOR = (255, 173, 92)
TEXT = (245, 246, 250)
MUTED = (170, 178, 192)

def draw(screen, font, small, env, message, mode):
    screen.fill(BG)
    title = font.render("Tic-Tac-Toe | Q-Learning Agent", True, TEXT)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 38)))
    sub = small.render(mode, True, MUTED)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, 68)))

    for i in (1, 2):
        pygame.draw.line(screen, GRID, (i * CELL, BOARD_TOP),
                         (i * CELL, BOARD_TOP + 3 * CELL), 5)
        pygame.draw.line(screen, GRID, (0, BOARD_TOP + i * CELL),
                         (WIDTH, BOARD_TOP + i * CELL), 5)

    for index, value in enumerate(env.board):
        if value == 0:
            continue
        row, col = divmod(index, 3)
        cx, cy = col * CELL + CELL // 2, BOARD_TOP + row * CELL + CELL // 2
        if value == 1:
            pad = 48
            pygame.draw.line(screen, X_COLOR, (cx-pad, cy-pad), (cx+pad, cy+pad), 12)
            pygame.draw.line(screen, X_COLOR, (cx+pad, cy-pad), (cx-pad, cy+pad), 12)
        else:
            pygame.draw.circle(screen, O_COLOR, (cx, cy), 58, 12)

    label = font.render(message, True, TEXT)
    screen.blit(label, label.get_rect(center=(WIDTH // 2, 660)))

def load_agent():
    path = Path(__file__).resolve().parent / "models" / "q_table.pkl"
    if not path.exists():
        raise SystemExit("No trained model found. Run: python train.py")
    return QLearningAgent.load(path)

def self_play(agent, games):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Trained RL agents playing Tic-Tac-Toe")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 28, bold=True)
    small = pygame.font.SysFont("arial", 18)
    env = TicTacToe()
    game_no, player = 1, 1
    done, winner = False, 0
    paused, next_game, last_move = False, False, 0
    message = "X (blue) vs O (orange) | SPACE pause | N next game"
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    next_game = True

        now = pygame.time.get_ticks()
        if next_game:
            game_no += 1
            if game_no > games:
                running = False
            else:
                env.reset()
                player, done, winner = 1, False, 0
                next_game, last_move = False, now
                message = f"Game {game_no}/{games}: X vs O"
        elif not paused and not done and now - last_move > 650:
            action = agent.choose_action(env.state(), env.legal_actions(), player=player)
            _, done, winner = env.step(action, player)
            player *= -1
            last_move = now
            if done:
                if winner == 1:
                    message = f"Game {game_no}: X wins"
                elif winner == -1:
                    message = f"Game {game_no}: O wins"
                else:
                    message = f"Game {game_no}: Draw"
        elif done and now - last_move > 1600:
            message = f"{message} | Press N for next game"

        if paused and not done:
            display_message = "PAUSED | SPACE resume | N next game"
        else:
            display_message = message
        draw(screen, font, small, env, display_message, f"Self-play | Game {game_no}/{games}")
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

def human_play(agent):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Play Tic-Tac-Toe against the trained RL agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 28, bold=True)
    small = pygame.font.SysFont("arial", 18)
    env = TicTacToe()
    human, ai = 1, -1  # Human is X, trained agent is O.
    message = "You are X (blue). Click a square to play."
    ai_turn = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    ai_turn = False
                    message = "New game. You are X (blue)."
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos
                if y >= BOARD_TOP and y < BOARD_TOP + 3 * CELL and not env.done and not ai_turn:
                    col, row = x // CELL, (y - BOARD_TOP) // CELL
                    action = int(row * 3 + col)
                    if action in env.legal_actions():
                        _, done, winner = env.step(action, human)
                        if done:
                            message = "You win!" if winner == human else "Draw!"
                        else:
                            ai_turn = True

        if ai_turn and not env.done:
            pygame.time.wait(350)
            action = agent.choose_action(env.state(), env.legal_actions(), player=ai)
            _, done, winner = env.step(action, ai)
            ai_turn = False
            if done:
                message = "Agent wins!" if winner == ai else "Draw!"

        draw(screen, font, small, env, message + " | R restart | ESC quit", "Human vs trained agent")
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["self", "human"], default="self")
    parser.add_argument("--games", type=int, default=5)
    args = parser.parse_args()
    trained_agent = load_agent()
    if args.mode == "self":
        self_play(trained_agent, max(1, args.games))
    else:
        human_play(trained_agent)
