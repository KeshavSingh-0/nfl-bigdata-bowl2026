import pygame
import csv
import math
import os
import time
from collections import defaultdict

# Paths to CSVs
INPUT_CSV = r"kaggle_file\train_input\input_2023_w01.csv"
OUTPUT_CSV = r"kaggle_file\train_output\output_2023_w01.csv"

# Constants
FIELD_WIDTH = 1200
FIELD_HEIGHT = 533
MARGIN = 50
SIDE_PANEL_WIDTH = 350
INFO_BOX_WIDTH = 250
WINDOW_WIDTH = FIELD_WIDTH + 2 * MARGIN + SIDE_PANEL_WIDTH + INFO_BOX_WIDTH
WINDOW_HEIGHT = FIELD_HEIGHT + 2 * MARGIN

FIELD_LENGTH_YARDS = 120.0
FIELD_WIDTH_YARDS = 53.3

FPS = 30
PLAYER_RADIUS = 20
STAR_SIZE = 5  
TRAIL_LENGTH = 0

# Colors
BLUE = (50, 150, 255)
RED = (255, 50, 50)
LIGHT_BLUE = (150, 200, 255)
LIGHT_RED = (255, 170, 170)
WHITE = (255, 255, 255)
GREEN = (40, 140, 40)
DARK_GREEN_ENDZONE = (0, 100, 0)
YELLOW_FADE = (255, 255, 150, 100)  # faded yellow (with alpha)
YELLOW_DEEP = (255, 220, 0)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
STAR_COLOR = YELLOW_DEEP

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("NFL Play Visualizer")

# Load NFL style font if available, fallback to Arial
try:
    font_path = pygame.font.match_font('nfl') or pygame.font.match_font('arial')
except Exception:
    font_path = None

if font_path:
    font = pygame.font.Font(font_path, 14)
    font_big = pygame.font.Font(font_path, 20)
    font_panel_title = pygame.font.Font(font_path, 18)
    font_endzone = pygame.font.Font(font_path, 36)
else:
    font = pygame.font.SysFont("Arial", 14)
    font_big = pygame.font.SysFont("Arial", 20)
    font_panel_title = pygame.font.SysFont("Arial", 18)
    font_endzone = pygame.font.SysFont("Arial", 36)

clock = pygame.time.Clock()

def to_screen_coords(x, y):
    sx = MARGIN + (x / FIELD_LENGTH_YARDS) * FIELD_WIDTH
    sy = MARGIN + FIELD_HEIGHT - (y / FIELD_WIDTH_YARDS) * FIELD_HEIGHT
    return int(sx), int(sy)

def draw_star(surface, x, y, size=STAR_SIZE, color=STAR_COLOR):
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

def draw_field(surface):
    surface.fill(GREEN)
    # Endzones
    left_x1 = MARGIN
    left_x2 = MARGIN + (10 / FIELD_LENGTH_YARDS) * FIELD_WIDTH
    right_x1 = MARGIN + (110 / FIELD_LENGTH_YARDS) * FIELD_WIDTH
    right_x2 = MARGIN + FIELD_WIDTH
    pygame.draw.rect(surface, DARK_GREEN_ENDZONE, (left_x1, MARGIN, left_x2 - left_x1, FIELD_HEIGHT))
    pygame.draw.rect(surface, DARK_GREEN_ENDZONE, (right_x1, MARGIN, right_x2 - right_x1, FIELD_HEIGHT))

    # NFL Big Data label centered & bigger
    text_left = font_endzone.render("NFL Big Data", True, WHITE)
    text_right = font_endzone.render("NFL Big Data", True, WHITE)
    # Left rotated CCW
    text_left_surf = pygame.transform.rotate(text_left, 90)
    surface.blit(text_left_surf, (left_x1 + (left_x2-left_x1)//2 - text_left_surf.get_width()//2, MARGIN + FIELD_HEIGHT//2 - text_left_surf.get_height()//2))
    # Right rotated CW
    text_right_surf = pygame.transform.rotate(text_right, -90)
    surface.blit(text_right_surf, (right_x2 - (right_x2 - right_x1)//2 - text_right_surf.get_width()//2, MARGIN + FIELD_HEIGHT//2 - text_right_surf.get_height()//2))

    # Yard lines every 10 yards
    for i in range(0, 121, 10):
        x = MARGIN + (i / FIELD_LENGTH_YARDS) * FIELD_WIDTH
        pygame.draw.line(surface, WHITE, (x, MARGIN), (x, MARGIN + FIELD_HEIGHT), 2)
        if 10 <= i <= 110:
            num_text = font.render(str(i), True, WHITE)
            surface.blit(num_text, (x - num_text.get_width()//2, MARGIN - 20))
            surface.blit(num_text, (x - num_text.get_width()//2, MARGIN + FIELD_HEIGHT + 5))

    # Sidelines
    pygame.draw.rect(surface, WHITE, (MARGIN, MARGIN, FIELD_WIDTH, FIELD_HEIGHT), 5)

def load_input_data(filename):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not os.path.exists(filename):
        print(f"Input file not found: {filename}")
        return data

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                gid = int(row["game_id"])
                pid = int(row["play_id"])
                nfl = int(row["nfl_id"])
                fid = int(row["frame_id"])
                x = float(row["x"])
                y = float(row["y"])
                player_to_predict = str(row["player_to_predict"]).lower() == "true"
                player_name = row["player_name"]
                player_position = row["player_position"]
                player_side = row["player_side"]
                player_role = row["player_role"]
                ball_land_x = float(row.get("ball_land_x", 0) or 0)
                ball_land_y = float(row.get("ball_land_y", 0) or 0)
            except Exception:
                continue

            player = {
                "game_id": gid,
                "play_id": pid,
                "nfl_id": nfl,
                "frame_id": fid,
                "x": x,
                "y": y,
                "player_to_predict": player_to_predict,
                "player_name": player_name,
                "player_position": player_position,
                "player_side": player_side,
                "player_role": player_role,
                "ball_land_x": ball_land_x,
                "ball_land_y": ball_land_y
            }
            data[gid][pid][fid].append(player)
    return data

def load_output_data(filename):
    out_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    if not os.path.exists(filename):
        print(f"Output file not found: {filename}")
        return out_data

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                gid = int(row["game_id"])
                pid = int(row["play_id"])
                nfl = int(row["nfl_id"])
                fid = int(row["frame_id"])
                x = float(row["x"])
                y = float(row["y"])
            except Exception:
                continue
            out_data[gid][pid][fid][nfl] = {"x": x, "y": y}
    return out_data

def draw_dotted_line(surf, color, start_pos, end_pos, width=2, dash_length=5, space_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dash_count = int(length // (dash_length + space_length))
    if dash_count == 0:
        dash_count = 1
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    for i in range(dash_count):
        start_x = x1 + (dash_length + space_length) * i * dx
        start_y = y1 + (dash_length + space_length) * i * dy
        end_x = start_x + dash_length * dx
        end_y = start_y + dash_length * dy
        pygame.draw.line(surf, color, (start_x, start_y), (end_x, end_y), width)

def main():
    input_data = load_input_data(INPUT_CSV)
    output_data = load_output_data(OUTPUT_CSV)

    game_ids = sorted(input_data.keys())
    for g in output_data.keys():
        if g not in game_ids:
            game_ids.append(g)
    game_ids = sorted(game_ids)

    if not game_ids:
        print("No game data loaded.")
        return

    plays_per_game = {gid: sorted(set(list(input_data[gid].keys()) + list(output_data[gid].keys()))) for gid in game_ids}

    current_game_idx = 0
    current_game_id = game_ids[current_game_idx]
    play_ids = plays_per_game[current_game_id]
    current_play_idx = 0
    current_play_id = play_ids[current_play_idx]

    current_frame = 1
    autoplay = False
    autoplay_speed = 8
    last_autoplay_time = time.time()

    # Store full star trails for all frames
    star_full_history = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    running = True
    hovered_player = None

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    current_play_idx = (current_play_idx + 1) % len(play_ids)
                    current_play_id = play_ids[current_play_idx]
                    current_frame = 1
                    star_full_history.clear()
                elif event.key == pygame.K_LEFT:
                    current_play_idx = (current_play_idx - 1) % len(play_ids)
                    current_play_id = play_ids[current_play_idx]
                    current_frame = 1
                    star_full_history.clear()
                elif event.key == pygame.K_UP:
                    current_frame += 1
                elif event.key == pygame.K_DOWN:
                    current_frame -= 1
                    if current_frame < 1:
                        current_frame = 1
                elif event.key == pygame.K_SPACE:
                    autoplay = not autoplay
                elif event.key == pygame.K_r:
                    current_frame = 1
                    star_full_history.clear()
                elif event.key == pygame.K_g:
                    current_game_idx = (current_game_idx + 1) % len(game_ids)
                    current_game_id = game_ids[current_game_idx]
                    play_ids = plays_per_game[current_game_id]
                    current_play_idx = 0
                    current_play_id = play_ids[current_play_idx]
                    current_frame = 1
                    star_full_history.clear()

        if autoplay and (time.time() - last_autoplay_time >= 1.0 / autoplay_speed):
            current_frame += 1
            last_autoplay_time = time.time()

        draw_field(screen)

        gid = current_game_id
        pid = current_play_id

        input_frames = input_data.get(gid, {}).get(pid, {})
        output_frames = output_data.get(gid, {}).get(pid, {})

        last_input_frame = max(input_frames.keys()) if input_frames else 0
        last_output_frame = max(output_frames.keys()) if output_frames else 0
        total_frames = last_input_frame + last_output_frame if (last_input_frame + last_output_frame) > 0 else 1

        if current_frame < 1:
            current_frame = 1
        if current_frame > total_frames:
            current_frame = total_frames

        in_output_mode = current_frame > last_input_frame
        players_to_draw = []

        if not in_output_mode:
            players_to_draw = input_frames.get(current_frame, [])
        else:
            out_frame = current_frame - last_input_frame
            last_input_meta = input_frames.get(last_input_frame, [])
            if not last_input_meta and input_frames:
                min_frame = min(input_frames.keys())
                last_input_meta = input_frames.get(min_frame, [])
            if last_input_meta:
                for meta in last_input_meta:
                    nid = meta.get("nfl_id")
                    if nid is None:
                        continue
                    predicted = output_frames.get(out_frame, {}).get(nid)
                    meta_copy = meta.copy()
                    if predicted:
                        meta_copy["x"] = predicted["x"]
                        meta_copy["y"] = predicted["y"]
                    players_to_draw.append(meta_copy)
            else:
                for nid, coords in output_frames.get(out_frame, {}).items():
                    players_to_draw.append({
                        "game_id": gid,
                        "play_id": pid,
                        "player_to_predict": False,
                        "nfl_id": nid,
                        "frame_id": current_frame,
                        "play_direction": "",
                        "player_name": f"NFL:{nid}",
                        "player_position": "",
                        "player_side": "Unknown",
                        "player_role": "",
                        "x": coords["x"],
                        "y": coords["y"],
                        "ball_land_x": 0.0,
                        "ball_land_y": 0.0
                    })

        # Update star full trail for star player (only one star per play assumed)
        star_player = None
        for p in players_to_draw:
            if p.get("player_to_predict", False):
                star_player = p
                break
        if star_player:
            nid_star = star_player.get("nfl_id")
            x, y = star_player.get("x"), star_player.get("y")
            star_full_history[gid][pid][nid_star].append((x, y))
            # Keep all points, no max length limit

        # Draw star player dotted line trail for entire history (all frames)
        if star_player:
            nid_star = star_player.get("nfl_id")
            full_trail_points = star_full_history[gid][pid][nid_star]
            if len(full_trail_points) > 1:
                points_px = [to_screen_coords(x, y) for x, y in full_trail_points]
                for i in range(len(points_px) - 1):
                    draw_dotted_line(screen, STAR_COLOR, points_px[i], points_px[i + 1], width=3, dash_length=8, space_length=6)

        # Draw all players (no trails except star)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        hovered_player = None

        for p in players_to_draw:
            x, y = p.get("x"), p.get("y")
            if x is None or y is None:
                continue
            sx, sy = to_screen_coords(float(x), float(y))
            side = str(p.get("player_side") or "").strip().lower()
            player_to_predict = p.get("player_to_predict", False)
            name = p.get("player_name") or ""
            position = p.get("player_position") or ""
            nfl_id = p.get("nfl_id")

            # Determine color and saturation
            if player_to_predict:
                color = BLUE if side == "offense" else RED
                # Star player remains saturated after pass
            else:
                if in_output_mode:
                    color = LIGHT_BLUE if side == "offense" else LIGHT_RED if side == "defense" else GRAY
                else:
                    color = BLUE if side == "offense" else RED if side == "defense" else GRAY

            pygame.draw.circle(screen, color, (sx, sy), PLAYER_RADIUS)

            # Position letter big on circle
            if position:
                pos_letter = position[0].upper()
                text_pos = font_big.render(pos_letter, True, WHITE)
                text_rect = text_pos.get_rect(center=(sx, sy))
                screen.blit(text_pos, text_rect)

            # Smaller star for starred player
            if player_to_predict:
                draw_star(screen, sx, sy, size=STAR_SIZE, color=STAR_COLOR)

            # Hover detection
            dist = math.hypot(mouse_x - sx, mouse_y - sy)
            if dist < PLAYER_RADIUS + 5:
                hovered_player = p

        # Ball landing circle
        last_input_meta = input_frames.get(last_input_frame, [])
        ball_land_x, ball_land_y = None, None
        for p in last_input_meta:
            blx = p.get("ball_land_x")
            bly = p.get("ball_land_y")
            if blx and bly:
                ball_land_x, ball_land_y = blx, bly
                break
        if ball_land_x is not None and ball_land_y is not None:
            bx, by = to_screen_coords(ball_land_x, ball_land_y)
            radius = 5
            ball_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            if in_output_mode:
                pygame.draw.circle(ball_surf, YELLOW_DEEP + (200,), (radius, radius), radius)
            else:
                pygame.draw.circle(ball_surf, YELLOW_FADE, (radius, radius), radius)
            screen.blit(ball_surf, (bx - radius, by - radius))

        # Info Box
        panel_x = MARGIN + FIELD_WIDTH + SIDE_PANEL_WIDTH + 20
        panel_y = MARGIN
        info_box_rect = pygame.Rect(panel_x, panel_y, INFO_BOX_WIDTH - 20, FIELD_HEIGHT)
        pygame.draw.rect(screen, DARK_GRAY, info_box_rect)
        title_text = font_panel_title.render("Dataset Info", True, WHITE)
        screen.blit(title_text, (panel_x + 10, panel_y + 10))
        num_games_text = font.render(f"Total Games: {len(game_ids)}", True, WHITE)
        screen.blit(num_games_text, (panel_x + 10, panel_y + 50))
        plays_for_current_game = plays_per_game.get(current_game_id, [])
        plays_count = len(plays_for_current_game)
        plays_text = font.render(f"Current Game: {current_game_id}", True, WHITE)
        screen.blit(plays_text, (panel_x + 10, panel_y + 80))
        plays_count_text = font.render(f"Number of Plays: {plays_count}", True, WHITE)
        screen.blit(plays_count_text, (panel_x + 10, panel_y + 110))

        # Side Panel - Players
        side_panel_x = MARGIN + FIELD_WIDTH + 10
        side_panel_y = MARGIN
        side_panel_rect = pygame.Rect(side_panel_x, side_panel_y, SIDE_PANEL_WIDTH, FIELD_HEIGHT)
        pygame.draw.rect(screen, DARK_GRAY, side_panel_rect)
        sp_title = font_panel_title.render("Players in Play", True, WHITE)
        screen.blit(sp_title, (side_panel_x + 10, side_panel_y + 10))

        # Players list from last input frame (defense + offense)
        player_list = last_input_meta if last_input_meta else players_to_draw

        # Remove duplicates by nfl_id (keep first)
        seen_nfls = set()
        filtered_players = []
        for p in player_list:
            nfl = p.get("nfl_id")
            if nfl in seen_nfls:
                continue
            seen_nfls.add(nfl)
            filtered_players.append(p)

        def player_sort_key(p):
            side_order = {"offense": 0, "defense": 1}
            side = p.get("player_side", "").lower()
            pos = p.get("player_position", "")
            name = p.get("player_name", "")
            return (side_order.get(side, 99), pos, name)

        filtered_players.sort(key=player_sort_key)

        y_offset = side_panel_y + 40
        line_height = 22
        max_lines = (FIELD_HEIGHT - 50) // line_height

        for i, p in enumerate(filtered_players[:max_lines]):
            side = str(p.get("player_side") or "").strip().lower()
            player_to_predict = p.get("player_to_predict", False)

            if player_to_predict:
                color = BLUE if side == "offense" else RED
            else:
                color = LIGHT_BLUE if side == "offense" else LIGHT_RED if side == "defense" else GRAY

            name = p.get("player_name") or "Unknown"
            pos = p.get("player_position") or ""
            role = p.get("player_role") or ""

            text_str = f"{name} | {pos} | {role}"
            text_surface = font.render(text_str, True, color)
            screen.blit(text_surface, (side_panel_x + 10, y_offset + i * line_height))

        # Hover box near mouse
        if hovered_player:
            info_str = f"{hovered_player.get('player_name','')} — {hovered_player.get('player_position','')} — {hovered_player.get('player_role','')}"
            text_bg = font.render(info_str, True, BLACK, WHITE)
            mouse_x, mouse_y = pygame.mouse.get_pos()
            screen.blit(text_bg, (mouse_x + 12, mouse_y + 12))

        # Controls & status top-left
        mode_text = "POST-PASS (PREDICTION)" if in_output_mode else "PRE-PASS (TRACKING)"
        info_text = f"Game: {gid} | Play: {pid} | Frame: {current_frame}/{total_frames} | {mode_text}"
        info_surf = font.render(info_text, True, WHITE)
        screen.blit(info_surf, (MARGIN, 10))

        controls = "LEFT/RIGHT: Change Play | UP/DOWN: Change Frame | SPACE: Play/Pause | R: Reset | G: Next Game"
        ctrl_surf = font.render(controls, True, WHITE)
        screen.blit(ctrl_surf, (MARGIN, FIELD_HEIGHT + MARGIN + 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()