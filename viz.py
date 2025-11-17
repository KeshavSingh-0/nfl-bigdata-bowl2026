import pygame
import csv
import math
from collections import defaultdict

# === Constants ===
WIDTH, HEIGHT = 1200, 533  # Scaled for 120 yards x 53.3 yards field
FIELD_LENGTH_YARDS = 120
FIELD_WIDTH_YARDS = 53.3
MARGIN = 50

BLUE = (50, 150, 255)
RED = (255, 50, 50)
WHITE = (255, 255, 255)
GREEN = (50, 180, 50)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

FPS = 30

pygame.init()
screen = pygame.display.set_mode((WIDTH + 2 * MARGIN, HEIGHT + 2 * MARGIN))
pygame.display.set_caption("NFL Play Visualizer")
font = pygame.font.SysFont('Arial', 16)
font_big = pygame.font.SysFont('Arial', 24, bold=True)

# Convert yards to pixels
def to_screen_coords(x, y):
    # Flip y to make top left as (0,0) and scale
    screen_x = MARGIN + (x / FIELD_LENGTH_YARDS) * WIDTH
    screen_y = MARGIN + HEIGHT - (y / FIELD_WIDTH_YARDS) * HEIGHT
    return int(screen_x), int(screen_y)

# Draw a star to highlight player_to_predict
def draw_star(surface, x, y, size=12, color=YELLOW):
    points = []
    for i in range(5):
        angle = i * (2 * math.pi / 5) - math.pi / 2
        x_outer = x + size * math.cos(angle)
        y_outer = y + size * math.sin(angle)
        points.append((x_outer, y_outer))
        angle_inner = angle + math.pi / 5
        x_inner = x + (size / 2) * math.cos(angle_inner)
        y_inner = y + (size / 2) * math.sin(angle_inner)
        points.append((x_inner, y_inner))
    pygame.draw.polygon(surface, color, points)

# Load CSV and organize data by game_id and play_id and frames
def load_data(filename):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Structure: data[game_id][play_id][frame_id] = list of players
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=[
            "game_id","play_id","player_to_predict","nfl_id","frame_id",
            "play_direction","absolute_yardline_number","player_name","player_height","player_weight",
            "player_birth_date","player_position","player_side","player_role","x","y","s","a","o","dir",
            "num_frames_output","ball_land_x","ball_land_y"
        ])
        
        for row in reader:
            # Convert numeric fields
            try:
                row['game_id'] = int(row['game_id'])
                row['play_id'] = int(row['play_id'])
                row['player_to_predict'] = row['player_to_predict'].lower() == 'true'
                row['nfl_id'] = int(row['nfl_id'])
                row['frame_id'] = int(row['frame_id'])
                row['absolute_yardline_number'] = float(row['absolute_yardline_number'])
                row['x'] = float(row['x'])
                row['y'] = float(row['y'])
                row['s'] = float(row['s'])
                row['a'] = float(row['a'])
                row['o'] = float(row['o'])
                row['dir'] = float(row['dir'])
                row['num_frames_output'] = int(row['num_frames_output'])
                row['ball_land_x'] = float(row['ball_land_x'])
                row['ball_land_y'] = float(row['ball_land_y'])
            except Exception as e:
                print("Skipping row due to conversion error:", e)
                continue
            data[row['game_id']][row['play_id']][row['frame_id']].append(row)
    
    return data

def draw_field(surface):
    surface.fill(GREEN)
    # Draw yard lines every 10 yards
    for i in range(0, 121, 10):
        x = MARGIN + (i / FIELD_LENGTH_YARDS) * WIDTH
        pygame.draw.line(surface, WHITE, (x, MARGIN), (x, HEIGHT + MARGIN), 2)
        # Draw yard numbers every 10 yards except end zones
        if 10 <= i <= 110:
            number_text = font.render(str(i), True, WHITE)
            surface.blit(number_text, (x - number_text.get_width()//2, HEIGHT + MARGIN + 5))
            surface.blit(number_text, (x - number_text.get_width()//2, MARGIN - 20))
    # Sidelines
    pygame.draw.rect(surface, WHITE, (MARGIN, MARGIN, WIDTH, HEIGHT), 5)

def main():
    clock = pygame.time.Clock()
    data = load_data('kaggle_file\\train\\input_2023_w01.csv')  # Replace with your filename

    # Sort game and play IDs for navigation
    game_ids = sorted(data.keys())
    if not game_ids:
        print("No data loaded.")
        return

    current_game_idx = 0
    current_play_ids = sorted(data[game_ids[current_game_idx]].keys())
    if not current_play_ids:
        print("No plays for current game.")
        return
    current_play_idx = 0

    current_frame = 1

    running = True
    hovered_player = None

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Left/right to change play
                if event.key == pygame.K_RIGHT:
                    current_play_idx += 1
                    if current_play_idx >= len(current_play_ids):
                        current_play_idx = 0
                    current_frame = 1
                elif event.key == pygame.K_LEFT:
                    current_play_idx -= 1
                    if current_play_idx < 0:
                        current_play_idx = len(current_play_ids) - 1
                    current_frame = 1
                # Up/down to change frame
                elif event.key == pygame.K_UP:
                    current_frame += 1
                elif event.key == pygame.K_DOWN:
                    current_frame -= 1
                    if current_frame < 1:
                        current_frame = 1

        screen.fill(GREEN)
        draw_field(screen)

        game_id = game_ids[current_game_idx]
        play_id = current_play_ids[current_play_idx]

        frames = data[game_id][play_id]
        if not frames:
            continue

        max_frame = max(frames.keys())
        if current_frame > max_frame:
            current_frame = max_frame

        players = frames.get(current_frame, [])

        mouse_x, mouse_y = pygame.mouse.get_pos()
        hovered_player = None

        # Draw players
        for p in players:
            
            # Init px and py
            px, py = to_screen_coords(p['x'], p['y'])
            color = BLUE if p['player_side'].lower() == 'offense' else RED


            # Highlight player_to_predict with star
            if p['player_to_predict']:
                draw_star(screen, px, py, size=15)


            # Draw Circles
            pygame.draw.circle(screen, color, (px, py), 20)
            
            
            # Draw position letters
            pos_text = font_big.render(p['player_position'], True, WHITE)
            pos_rect = pos_text.get_rect(center=(px, py))
            screen.blit(pos_text, pos_rect)


            # Detect hover
            dist = math.hypot(mouse_x - px, mouse_y - py)
            if dist <= 20:
                hovered_player = p
            
            
            
            

            
    

        # Draw football position - ball_land_x, ball_land_y from any player in the frame (all the same per play)
        if players:
            ball_x = players[0]['ball_land_x']
            ball_y = players[0]['ball_land_y']
            bx, by = to_screen_coords(ball_x, ball_y)
            pygame.draw.circle(screen, BLACK, (bx, by), 10)
            pygame.draw.circle(screen, YELLOW, (bx, by), 6)

        # Display info for hovered player
        if hovered_player:
            info_text = f"{hovered_player['player_name']} - {hovered_player['player_position']} - {hovered_player['player_role']}"
            text_surface = font.render(info_text, True, BLACK, WHITE)
            screen.blit(text_surface, (mouse_x + 15, mouse_y))

        # Draw current play/frame info
        play_info = f"Game: {game_id} | Play: {play_id} | Frame: {current_frame}/{max_frame}"
        info_surface = font.render(play_info, True, WHITE)
        screen.blit(info_surface, (MARGIN, 10))

        controls_info = "Arrows Left/Right: Change Play | Arrows Up/Down: Change Frame"
        controls_surface = font.render(controls_info, True, WHITE)
        screen.blit(controls_surface, (MARGIN, HEIGHT + MARGIN + 25))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

