"""
TRON LIGHT CYCLE GAME
=====================
AI Algorithm: A* Search (from AI Lesson Plan - Module 3: Informed Heuristics Strategies)
  - The enemy agent uses A* to find the optimal path to intercept the player,
    avoiding trail collisions. It re-plans every few frames (agentic loop).

Controls:
  Arrow Keys / WASD - Move player
  R                 - Restart after game over
  ESC               - Quit

Dependencies: pip install pygame
"""

import pygame
import sys
import heapq
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CELL       = 16           # pixels per grid cell
COLS       = 60
ROWS       = 40
WIDTH      = COLS * CELL
HEIGHT     = ROWS * CELL
FPS        = 15

# Tron neon colour palette
BG_COLOR        = (2,   4,  16)
GRID_COLOR      = (0,  20,  40)
PLAYER_COLOR    = (0, 200, 255)   # cyan
PLAYER_TRAIL    = (0,  80, 140)
ENEMY_COLOR     = (255, 50,  80)  # red-orange
ENEMY_TRAIL     = (120, 20,  30)
TEXT_COLOR      = (200, 240, 255)
GLOW_COLOR      = (0, 150, 255)
WIN_COLOR       = (0, 255, 150)
LOSE_COLOR      = (255, 60,  60)

# Directions
UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)
DIRS  = [UP, DOWN, LEFT, RIGHT]

def opposite(d):
    return (-d[0], -d[1])


# ─────────────────────────────────────────────
# A* SEARCH
# ─────────────────────────────────────────────
# f(n) = g(n) + h(n)
# g(n) = cost from start to n (steps taken)
# h(n) = Manhattan distance heuristic to goal
# The heuristic is admissible: never overestimates
# (each cell costs 1, Manhattan ≤ actual path).

def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    """Manhattan distance — admissible & consistent heuristic."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(start: Tuple[int,int],
          goal:  Tuple[int,int],
          blocked: set) -> Optional[List[Tuple[int,int]]]:
    """
    A* Search on a 2-D grid.
    Returns the shortest path (list of cells) from start to goal,
    or None if no path exists.
    """
    # Priority queue: (f, g, node, path)
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, [start]))
    closed = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current == goal:
            return path

        if current in closed:
            continue
        closed.add(current)

        for dx, dy in DIRS:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
            if nx < 0 or ny < 0 or nx >= COLS or ny >= ROWS:
                continue
            if neighbor in blocked:
                continue

            g_next = g + 1
            f_next = g_next + heuristic(neighbor, goal)
            heapq.heappush(open_list, (f_next, g_next, neighbor, path + [neighbor]))

    return None   # no path found


# ─────────────────────────────────────────────
# FLOOD FILL (survival heuristic)
# ─────────────────────────────────────────────
def flood_fill_count(start: Tuple[int,int], blocked: set) -> int:
    """BFS flood fill – counts reachable cells from start."""
    visited = {start}
    queue   = [start]
    while queue:
        x, y = queue.pop()
        for dx, dy in DIRS:
            nx, ny = x+dx, y+dy
            n = (nx, ny)
            if n not in visited and n not in blocked \
               and 0 <= nx < COLS and 0 <= ny < ROWS:
                visited.add(n)
                queue.append(n)
    return len(visited)


# ─────────────────────────────────────────────
# CYCLE (player / enemy)
# ─────────────────────────────────────────────
@dataclass
class Cycle:
    pos:        Tuple[int,int]
    direction:  Tuple[int,int]
    trail:      set = field(default_factory=set)
    alive:      bool = True

    def __post_init__(self):
        self.trail.add(self.pos)

    def next_pos(self, d=None) -> Tuple[int,int]:
        d = d or self.direction
        return (self.pos[0]+d[0], self.pos[1]+d[1])

    def move(self):
        nxt = self.next_pos()
        self.pos = nxt
        self.trail.add(nxt)

    def in_bounds(self, pos=None) -> bool:
        x, y = pos or self.pos
        return 0 <= x < COLS and 0 <= y < ROWS


# ─────────────────────────────────────────────
# AGENTIC ENEMY AI
# ─────────────────────────────────────────────
class EnemyAgent:
    """
    Agentic AI loop:
      PERCEIVE → PLAN (A*) → ACT
    The agent re-plans every REPLAN_INTERVAL frames.
    Goal: intercept the player's *predicted future position*.
    Fallback: if intercepting is blocked, use flood-fill
    survival strategy (pick direction with most open space).
    """
    REPLAN_INTERVAL = 3   # replan every N game ticks

    def __init__(self, enemy: Cycle, player: Cycle):
        self.enemy  = enemy
        self.player = player
        self.plan:  List[Tuple[int,int]] = []
        self.ticks  = 0

    def _blocked(self) -> set:
        return self.enemy.trail | self.player.trail

    def _predict_player(self, steps=4) -> Tuple[int,int]:
        """Predict where player will be in `steps` moves (straight-line)."""
        px, py = self.player.pos
        dx, dy = self.player.direction
        for _ in range(steps):
            nx, ny = px+dx, py+dy
            if 0 <= nx < COLS and 0 <= ny < ROWS:
                px, py = nx, ny
        return (px, py)

    def decide(self) -> Tuple[int,int]:
        """
        Agentic decision:
        1. Try A* to intercept predicted player position
        2. Fallback: A* to current player position
        3. Fallback: flood-fill survival (pick most open direction)
        """
        self.ticks += 1
        blocked = self._blocked()

        # Re-plan on interval or if plan exhausted
        if self.ticks % self.REPLAN_INTERVAL == 0 or not self.plan:
            goal = self._predict_player(steps=5)
            path = astar(self.enemy.pos, goal, blocked)
            if path is None:
                path = astar(self.enemy.pos, self.player.pos, blocked)
            if path and len(path) > 1:
                self.plan = path[1:]   # skip current position
            else:
                self.plan = []

        # Follow plan
        if self.plan:
            nxt = self.plan.pop(0)
            dx = nxt[0] - self.enemy.pos[0]
            dy = nxt[1] - self.enemy.pos[1]
            chosen = (dx, dy)
            # Safety check: don't walk into a wall
            nx, ny = self.enemy.next_pos(chosen)
            if (nx, ny) not in blocked and self.enemy.in_bounds((nx, ny)):
                return chosen

        # Flood-fill survival fallback
        return self._survival_direction(blocked)

    def _survival_direction(self, blocked: set) -> Tuple[int,int]:
        """Choose direction that maximises reachable open space."""
        best_dir   = self.enemy.direction
        best_space = -1
        opp = opposite(self.enemy.direction)

        for d in DIRS:
            if d == opp:
                continue
            nx, ny = self.enemy.next_pos(d)
            if not (0 <= nx < COLS and 0 <= ny < ROWS):
                continue
            if (nx, ny) in blocked:
                continue
            space = flood_fill_count((nx, ny), blocked | {(nx, ny)})
            if space > best_space:
                best_space = space
                best_dir   = d

        return best_dir


# ─────────────────────────────────────────────
# GLOW DRAWING HELPER
# ─────────────────────────────────────────────
def draw_glow_cell(surface, color, x, y, cell, glow_r=3):
    cx, cy = x*cell + cell//2, y*cell + cell//2
    # Soft outer glow
    glow_surf = pygame.Surface((cell*4, cell*4), pygame.SRCALPHA)
    alpha = 40
    for r in range(glow_r, 0, -1):
        pygame.draw.circle(glow_surf, (*color, alpha),
                           (cell*2, cell*2), cell//2 + r*3)
        alpha += 20
    surface.blit(glow_surf, (cx - cell*2, cy - cell*2))
    # Core
    pygame.draw.rect(surface, color, (x*cell+1, y*cell+1, cell-2, cell-2))

def draw_trail_cell(surface, color, x, y, cell):
    pygame.draw.rect(surface, color, (x*cell+2, y*cell+2, cell-4, cell-4))


# ─────────────────────────────────────────────
# MAIN GAME
# ─────────────────────────────────────────────
class TronGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TRON  ·  Light Cycle  ·  A* AI")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()
        self.font_big   = pygame.font.SysFont("Courier New", 48, bold=True)
        self.font_small = pygame.font.SysFont("Courier New", 22, bold=True)
        self.font_tiny  = pygame.font.SysFont("Courier New", 14)
        self.reset()

    def reset(self):
        # Player starts left-center heading right
        self.player = Cycle(pos=(10, ROWS//2), direction=RIGHT)
        # Enemy starts right-center heading left
        self.enemy  = Cycle(pos=(COLS-10, ROWS//2), direction=LEFT)
        self.agent  = EnemyAgent(self.enemy, self.player)
        self.state  = "playing"   # "playing" | "win" | "lose"
        self.score  = 0
        self.frame  = 0

    def handle_input(self):
        keys = pygame.key.get_pressed()
        d = self.player.direction
        if (keys[pygame.K_UP]    or keys[pygame.K_w]) and d != DOWN:
            self.player.direction = UP
        elif (keys[pygame.K_DOWN]  or keys[pygame.K_s]) and d != UP:
            self.player.direction = DOWN
        elif (keys[pygame.K_LEFT]  or keys[pygame.K_a]) and d != RIGHT:
            self.player.direction = LEFT
        elif (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and d != LEFT:
            self.player.direction = RIGHT

    def update(self):
        if self.state != "playing":
            return

        # AI decision (agentic loop)
        new_dir = self.agent.decide()
        self.enemy.direction = new_dir

        # Move
        blocked = self.player.trail | self.enemy.trail
        p_nxt = self.player.next_pos()
        e_nxt = self.enemy.next_pos()

        p_crash = (not self.player.in_bounds(p_nxt)) or (p_nxt in blocked)
        e_crash = (not self.enemy.in_bounds(e_nxt))  or (e_nxt in blocked)
        head_on = (p_nxt == e_nxt)

        if p_crash and e_crash:
            self.state = "draw"
        elif p_crash or head_on:
            self.state = "lose"
        elif e_crash:
            self.state = "win"
            self.score += 1
        else:
            self.player.move()
            self.enemy.move()
            self.frame += 1

    def draw_grid(self):
        for x in range(0, WIDTH, CELL):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

    def draw_cycles(self):
        # Player trail
        for (x, y) in self.player.trail:
            if (x, y) != self.player.pos:
                draw_trail_cell(self.screen, PLAYER_TRAIL, x, y, CELL)
        # Enemy trail
        for (x, y) in self.enemy.trail:
            if (x, y) != self.enemy.pos:
                draw_trail_cell(self.screen, ENEMY_TRAIL, x, y, CELL)
        # Cycle heads
        draw_glow_cell(self.screen, PLAYER_COLOR, *self.player.pos, CELL)
        draw_glow_cell(self.screen, ENEMY_COLOR,  *self.enemy.pos,  CELL)

    def draw_hud(self):
        # Title
        t = self.font_small.render("TRON  LIGHT  CYCLE", True, GLOW_COLOR)
        self.screen.blit(t, (10, 8))

        # Legend
        p_lbl = self.font_tiny.render("▶ PLAYER (You)", True, PLAYER_COLOR)
        e_lbl = self.font_tiny.render("▶ ENEMY (A* AI)", True, ENEMY_COLOR)
        self.screen.blit(p_lbl, (10, HEIGHT - 36))
        self.screen.blit(e_lbl, (10, HEIGHT - 20))

        # Score / frame
        sc = self.font_small.render(f"WINS: {self.score}", True, TEXT_COLOR)
        self.screen.blit(sc, (WIDTH - 140, 8))

        fr = self.font_tiny.render(f"FRAME {self.frame}", True, GRID_COLOR)
        self.screen.blit(fr, (WIDTH//2 - 35, 10))

    def draw_overlay(self):
        if self.state == "playing":
            return
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        if self.state == "win":
            msg, color = "YOU WIN!", WIN_COLOR
        elif self.state == "lose":
            msg, color = "YOU LOSE!", LOSE_COLOR
        else:
            msg, color = "DRAW!", TEXT_COLOR

        txt = self.font_big.render(msg, True, color)
        self.screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 60))
        sub = self.font_small.render("Press R to restart  |  ESC to quit", True, TEXT_COLOR)
        self.screen.blit(sub, (WIDTH//2 - sub.get_width()//2, HEIGHT//2 + 10))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if event.key == pygame.K_r:
                        self.reset()

            self.handle_input()
            self.update()

            self.screen.fill(BG_COLOR)
            self.draw_grid()
            self.draw_cycles()
            self.draw_hud()
            self.draw_overlay()
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    TronGame().run()