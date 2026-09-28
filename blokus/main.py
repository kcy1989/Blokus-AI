"""Entry point: pygame init, event loop, state driver."""
import random
import sys

import pygame

from ui import UI, load_cjk_font_path
from game import Game


def main():
    pygame.init()
    pygame.display.set_caption("單人 BLOKUS")
    clock = pygame.time.Clock()
    ui = UI(None, load_cjk_font_path())
    ui.game = Game(random.Random())
    running = True
    while running:
        events = pygame.event.get()
        ui.tick(events)
        if not ui.running:
            running = False
        clock.tick(60)
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
