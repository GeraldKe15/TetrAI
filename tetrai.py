import pygame
import random
import sys
import copy


class Tetrai:
    WIDTH, HEIGHT = 300, 600
    BLOCK_SIZE = 30
    GRID_WIDTH = WIDTH // BLOCK_SIZE
    GRID_HEIGHT = (HEIGHT) // BLOCK_SIZE
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    COLORS = [(1, 237, 250), (254, 251, 52), (120, 37, 111),
              (46, 46, 132), (255, 145, 12), (234, 20, 28), (83, 218, 63)]

    SHAPES = [
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1]]
    ]

    '''
    Intializes Pygame

    Arguments:
        self

    '''

    def __init__(self):

        pygame.init()
        pygame.display.set_caption("TetrAI")
        self.screen = pygame.display.set_mode((Tetrai.WIDTH, Tetrai.HEIGHT))
        self.reset()

    '''
    Resets Pygame

    Arugments:
        self
    '''

    def reset(self):
        self.clock = pygame.time.Clock()
        self.grid = [
            [0] * Tetrai.GRID_WIDTH for _ in range(Tetrai.GRID_HEIGHT)]
        self.currshape_index = random.randint(0, 6)
        self.current_shape = Tetrai.SHAPES[self.currshape_index]
        self.current_x, self.current_y = Tetrai.GRID_WIDTH // 2 - 1, 0
        self.score = 0

    '''
    Displays the score on Pygame

    Arguments:
        self
        score: the score
    '''

    def display_score(self, score):
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, Tetrai.WHITE)
        self.screen.blit(score_text, (5, 5))

    '''
    Displays the game over screen with options to play again or quit

    Arugments:
        self
    '''

    def game_over_popup(self):
        game_over_font = pygame.font.Font(None, 50)
        score_font = pygame.font.Font(None, 36)

        translucency = (0, 0, 0, 128)
        self.screen.fill(translucency)

        game_over_text = game_over_font.render("GAME OVER", True, Tetrai.WHITE)
        score_text = score_font.render(
            f"Score: {self.score}", True, Tetrai.WHITE)
        play_again_text = score_font.render(
            "Press P to Play Again", True, Tetrai.WHITE)
        exit_text = score_font.render("Press Q to Quit", True, Tetrai.WHITE)

        self.screen.blit(game_over_text, (Tetrai.WIDTH //
                         2 - 105, Tetrai.HEIGHT // 2 - 50))
        self.screen.blit(score_text, (Tetrai.WIDTH //
                         2 - 50, Tetrai.HEIGHT // 2))
        self.screen.blit(play_again_text, (Tetrai.WIDTH //
                         2 - 120, Tetrai.HEIGHT // 2 + 50))
        self.screen.blit(exit_text, (Tetrai.WIDTH // 2 -
                         90, Tetrai.HEIGHT // 2 + 80))

        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        return True  # play again
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    '''
    Returns the score

    Arguments:
        self
    '''

    def get_score(self):
        return self.score

    '''
    Draws the board

    Arguments:
        self
    '''

    def draw_board(self):
        self.screen.fill(Tetrai.BLACK)
        self.draw_grid()
        self.draw_shape(self.current_shape, self.current_x * Tetrai.BLOCK_SIZE,
                        self.current_y * Tetrai.BLOCK_SIZE, Tetrai.COLORS[self.currshape_index])
        for row in range(Tetrai.GRID_HEIGHT):
            for col in range(Tetrai.GRID_WIDTH):
                if self.grid[row][col]:
                    self.draw_shape(Tetrai.SHAPES[-1], col * Tetrai.BLOCK_SIZE, row *
                                    Tetrai.BLOCK_SIZE, Tetrai.COLORS[-1*(self.grid[row][col])-1])

        self.draw_grid()
        self.display_score(self.score)

        pygame.display.update()
        self.clock.tick(500)

    '''
    Draws the grid

    Arguments:
        self
    '''

    def draw_grid(self):
        for x in range(0, Tetrai.WIDTH, Tetrai.BLOCK_SIZE):
            pygame.draw.line(self.screen, Tetrai.WHITE,
                             (x, 0+Tetrai.BLOCK_SIZE), (x, Tetrai.HEIGHT))
        for y in range(0, Tetrai.HEIGHT, Tetrai.BLOCK_SIZE):
            pygame.draw.line(self.screen, Tetrai.WHITE,
                             (0, y), (Tetrai.WIDTH, y))

    '''
    Draws the chosen shape

    Arguments:
        shape: the chosen shape
        x: x position
        y: y position
        color: the color of the shape

    '''

    def draw_shape(self, shape, x, y, color):
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col]:
                    pygame.draw.rect(self.screen, color, (x + col * Tetrai.BLOCK_SIZE,
                                                          y + row * Tetrai.BLOCK_SIZE, Tetrai.BLOCK_SIZE, Tetrai.BLOCK_SIZE))

    '''
    Returns the width of the piece

    Arguments:
        self
        shape: the shape
    '''

    def get_piece_width(self, shape):
        return len(shape[0])

    '''
    Returns the rotated shape 90 degrees clockwise

    Arguments:
        self
        shape: the shape
    '''

    def rotate_shape(self, shape):
        return list(map(list, zip(*shape[::-1])))

    '''
    Returns True if there is a collision

    Arguemnts:
        self
        shape: the shape
        x: x position
        y: y positions
        grid: the board
    '''

    def check_collision(self, shape, x, y, grid):
        for row in range(len(shape)):
            for col in range(len(shape[0])):
                if shape[row][col]:
                    if x + col < 0 or x + col >= Tetrai.GRID_WIDTH or y + row >= Tetrai.GRID_HEIGHT or grid[y + row][x + col]:
                        return True
        return False

    '''
    Returns the number of cleared rows

    Arguments:
        self
        grid: the board
    '''

    def clear_rows(self, grid):
        full_rows = [i for i, row in enumerate(grid) if all(row)]
        for row in full_rows:
            grid.pop(row)
            grid.insert(0, [0] * Tetrai.GRID_WIDTH)
        return len(full_rows)

    '''
    Returns the bumpiness (heuristic) of the stack on the board

    Arguments:
        self
        board: the board

    '''

    def bumpiness(self, board):
        transposed = list(zip(*board))
        heights = []
        for x in transposed:
            col = list(x)
            i = 0
            while i < len(board) and col[i] == 0:
                i += 1
            heights.append(i)
        bumpiness = sum(abs(heights[i] - heights[i + 1])
                        for i in range(len(heights) - 1))
        return bumpiness

    '''
    Returns the heights (heuristic) of the stack on the board

    Arguments:
        self
        board: the board
    '''

    def height(self, board):
        heights = [0] * Tetrai.GRID_WIDTH

        for col in range(Tetrai.GRID_WIDTH):
            for row in range(Tetrai.GRID_HEIGHT):
                if board[row][col] != 0:
                    heights[col] = Tetrai.GRID_HEIGHT - row
                    break

        sum_heights = sum(heights)
        return sum_heights

    '''
    Returns the number of completed lines (heuristic) on the board

    Arguments:
        self
        board: the board
    '''

    def complete_lines(self, board):
        full_rows = [i for i, row in enumerate(board) if all(row)]
        return len(full_rows)

    '''
    Returns the number of holes of the stack on the board

    Arguments:
        self
        board: the board
    '''

    def holes(self, board):
        num_holes = 0
        for col in range(Tetrai.GRID_WIDTH):
            block_found = False
            for row in range(Tetrai.GRID_HEIGHT):
                if board[row][col] != 0:
                    block_found = True
                elif block_found == True:
                    num_holes += 1
        return num_holes

    '''
    Returns the current state of the board in terms of the heuristics (bumpiness,
    height, holes, and completed lines)

    Arguments:
        self
        board: the board
    '''

    def get_state(self, board):
        bump = self.bumpiness(board)
        sum_height = self.height(board)
        cleared = self.complete_lines(board)
        holes = self.holes(board)

        # assigns the greatest weight on the 'cleared rows' heuristic
        cleared = cleared ** 3

        return [bump, sum_height, cleared, holes]

    '''
    Returns a deep copy of the board after checking whether a piece can be placed

    Arguments:
        self
        x: x position
        y: y position
        shape_index: index of the chosen shape
    '''

    def check_possible_piece(self, x, y, shape_index):
        grid = copy.deepcopy(self.grid)
        for i, j in Tetrai.SHAPES[shape_index]:
            grid[j + y][x + i] = (-1*(shape_index+1))
        return grid

    '''
    Returns the states reach possible rotation of a piece at each x position

    Arguments:
        self
    '''

    def get_next_states(self):
        next_states = []

        shape_index = self.currshape_index
        rotations = 1

        if shape_index == 0 or shape_index == 5 or shape_index == 6:
            rotations = 2
        elif shape_index == 1:
            rotations = 1
        else:
            rotations = 4

        curr_shape = self.current_shape
        width = self.get_piece_width(curr_shape)

        for rotation in range(rotations):
            for x in range(0, Tetrai.GRID_WIDTH - width+1):
                y = 0
                collision = self.check_collision(curr_shape, x, y+1, self.grid)
                # find lowest row that doesnt result in collision
                while not collision:
                    y += 1
                    collision = self.check_collision(
                        curr_shape, x, y+1, self.grid)

                new_grid = copy.deepcopy(self.grid)

                # simulate the move on the copied grid
                for row in range(len(curr_shape)):
                    for col in range(len(curr_shape[0])):
                        if curr_shape[row][col]:

                            new_grid[y + row][x + col] = - \
                                1*(self.currshape_index+1)

                state = self.get_state(new_grid)
                next_states.append(((x, y), rotation, state))

            rotated_shape = self.rotate_shape(curr_shape)
            curr_shape = rotated_shape

        return next_states

    '''
    Plays TetrAI

    Arguments:
        self
        move_action: the move action
        rotate_action: the rotate action
    '''

    def play(self, move_action, rotate_action, render=True):
        current_shape = self.current_shape

        for _ in range(rotate_action):
            current_shape = self.rotate_shape(current_shape)

        settled = False
        x, y = move_action
        r = 0

        while not settled:
            if self.current_x < x:
                self.current_x += 1
            elif self.current_x > x:
                self.current_x -= 1
            else:
                self.current_x = x

            if r < rotate_action:
                self.current_shape = self.rotate_shape(self.current_shape)
                r += 1

            if self.current_y < y:
                self.current_y += 1
            else:
                settled = True
                self.current_shape = current_shape
                for row in range(len(current_shape)):
                    for col in range(len(current_shape[0])):

                        if current_shape[row][col]:
                            self.grid[self.current_y + row][self.current_x +
                                                            col] = (-1*(self.currshape_index+1))
                            if self.current_y + row <= 0:
                                return self.score, True  # game over!
                cleared = self.clear_rows(self.grid)
                self.score += (cleared**2)
                self.currshape_index = random.randint(0, 6)
                self.current_shape = Tetrai.SHAPES[self.currshape_index]
                self.current_x, self.current_y = Tetrai.GRID_WIDTH // 2 - 1, 0
            if render:
                self.draw_board()

        return self.score, False
