import os
import subprocess
import tempfile
from pathlib import Path

import chess
import chess.pgn
from PIL import Image, ImageDraw, ImageFont

BOARD_SIZE = 480
FOOTER_HEIGHT = 108
SQUARE_SIZE = BOARD_SIZE // 8
FRAME_WIDTH = BOARD_SIZE
FRAME_HEIGHT = BOARD_SIZE + FOOTER_HEIGHT

LIGHT_BLUE = (214, 232, 255)
DARK_BLUE = (67, 116, 166)
FRAME_BG = (12, 26, 43)
FOOTER_BG = (15, 32, 52)
FOOTER_RULE = (76, 132, 190)
FRAME_RULE = (33, 74, 116)
TEXT_PRIMARY = (239, 246, 255)
TEXT_MUTED = (168, 191, 214)
WHITE_PIECE = (250, 252, 255)
BLACK_PIECE = (16, 24, 32)
WHITE_STROKE = (29, 48, 68)
BLACK_STROKE = (231, 240, 252)

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]

PIECE_CODEPOINTS = {
    chess.KING: 0x265A,
    chess.QUEEN: 0x265B,
    chess.ROOK: 0x265C,
    chess.BISHOP: 0x265D,
    chess.KNIGHT: 0x265E,
    chess.PAWN: 0x265F,
}


def _load_font(size):
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default(size=size)


PIECE_FONT = _load_font(54)
META_FONT = _load_font(20)
SMALL_FONT = _load_font(16)


def save_game_pgn(move_stack, white_name, black_name, result, game_num, opponent_type):
    """Save structural moves into a standardized PGN text file."""
    game = chess.pgn.Game()
    game.headers["Event"] = f"Transformer Benchmark Match - Game #{game_num}"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Comment"] = f"Opponent: {opponent_type}"

    node = game
    for move in move_stack:
        node = node.add_main_variation(move)

    os.makedirs("exports/pgns", exist_ok=True)
    filename = f"exports/pgns/game_{game_num}_{opponent_type}.pgn"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(game))
    return filename


def _parse_game_number(event):
    marker = "Game #"
    if marker not in event:
        return ""
    return event.split(marker, 1)[1].strip()


def metadata_from_game(game, source_path=None):
    headers = game.headers
    comment = headers.get("Comment", "")
    opponent = comment.replace("Opponent:", "").strip() if "Opponent:" in comment else ""
    return {
        "event": headers.get("Event", "Transformer Benchmark Match"),
        "game_num": _parse_game_number(headers.get("Event", "")),
        "white": headers.get("White", "?"),
        "black": headers.get("Black", "?"),
        "result": headers.get("Result", "*"),
        "opponent": opponent,
        "source": Path(source_path).name if source_path else "",
    }


def _piece_symbol(piece):
    return chr(PIECE_CODEPOINTS[piece.piece_type])


def _draw_centered_text(draw, xyxy, text, font, fill, stroke_fill=None, stroke_width=0):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = xyxy[0] + ((xyxy[2] - xyxy[0]) - width) / 2 - bbox[0]
    y = xyxy[1] + ((xyxy[3] - xyxy[1]) - height) / 2 - bbox[1]
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def _format_side_name(name):
    return "Model" if name == "Model" else name.replace("_", " ")


def _footer_metadata(metadata, move_index, total_moves, last_move):
    game_num = metadata.get("game_num") or "?"
    opponent = metadata.get("opponent") or "unknown"
    white = _format_side_name(metadata.get("white", "?"))
    black = _format_side_name(metadata.get("black", "?"))
    result = metadata.get("result", "*")
    move_text = f"Ply {move_index}/{total_moves}"
    if last_move:
        move_text += f" | Last move: {last_move}"
    else:
        move_text += " | Initial position"
    return {
        "match": f"Match #{game_num}",
        "summary": f"Opponent: {opponent} | Result: {result}",
        "white": white,
        "black": black,
        "move": move_text,
    }


def _text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _draw_piece_swatch(draw, x, y, fill, outline):
    draw.rounded_rectangle(
        (x, y, x + 22, y + 22),
        radius=5,
        fill=fill,
        outline=outline,
        width=2,
    )


def _draw_centered_line(draw, y, text, font, fill):
    width = _text_width(draw, text, font)
    draw.text(((FRAME_WIDTH - width) / 2, y), text, font=font, fill=fill)


def _draw_footer_header(draw, y, footer):
    _draw_piece_swatch(draw, 18, y, BLACK_PIECE, BLACK_STROKE)
    draw.text((48, y + 2), footer["black"], font=SMALL_FONT, fill=TEXT_PRIMARY)

    _draw_centered_line(draw, y + 2, footer["match"], SMALL_FONT, TEXT_MUTED)

    white_width = _text_width(draw, footer["white"], SMALL_FONT)
    swatch_x = FRAME_WIDTH - 40
    text_x = swatch_x - 10 - white_width
    draw.text((text_x, y + 2), footer["white"], font=SMALL_FONT, fill=TEXT_PRIMARY)
    _draw_piece_swatch(draw, swatch_x, y, WHITE_PIECE, WHITE_STROKE)


def _render_frame(board, metadata, move_index, total_moves, last_move=None):
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), FRAME_BG)
    draw = ImageDraw.Draw(img)

    for rank in range(8):
        for file_idx in range(8):
            display_rank = 7 - rank
            x0 = file_idx * SQUARE_SIZE
            y0 = display_rank * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            square_color = LIGHT_BLUE if (rank + file_idx) % 2 == 0 else DARK_BLUE
            draw.rectangle((x0, y0, x1, y1), fill=square_color)

            piece = board.piece_at(chess.square(file_idx, rank))
            if piece is None:
                continue

            fill = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
            stroke = WHITE_STROKE if piece.color == chess.WHITE else BLACK_STROKE
            _draw_centered_text(
                draw,
                (x0, y0 - 2, x1, y1 + 2),
                _piece_symbol(piece),
                PIECE_FONT,
                fill,
                stroke_fill=stroke,
                stroke_width=1,
            )

    footer_top = BOARD_SIZE
    draw.rectangle((0, footer_top, FRAME_WIDTH, FRAME_HEIGHT), fill=FOOTER_BG)
    draw.rectangle((0, footer_top, FRAME_WIDTH, footer_top + 3), fill=FOOTER_RULE)
    draw.rectangle((0, 0, FRAME_WIDTH - 1, FRAME_HEIGHT - 1), outline=FRAME_RULE, width=2)
    draw.rectangle((0, 0, FRAME_WIDTH - 1, BOARD_SIZE), outline=FRAME_RULE, width=2)

    footer = _footer_metadata(metadata, move_index, total_moves, last_move)
    _draw_footer_header(draw, footer_top + 15, footer)
    _draw_centered_line(draw, footer_top + 45, footer["summary"], META_FONT, TEXT_PRIMARY)
    _draw_centered_line(draw, footer_top + 74, footer["move"], SMALL_FONT, TEXT_MUTED)

    return img


def _render_game_frames(move_stack, metadata=None):
    board = chess.Board()
    metadata = metadata or {}
    frames = []
    total_moves = len(move_stack)

    frames.append(_render_frame(board, metadata, 0, total_moves))
    for move_index, move in enumerate(move_stack, start=1):
        board.push(move)
        frames.append(_render_frame(board, metadata, move_index, total_moves, move.uci()))

    return frames


def generate_game_gif(move_stack, filename_label, duration_per_move=1.0, metadata=None):
    """Compile a game move history into an animated GIF artifact."""
    frames = _render_game_frames(move_stack, metadata)
    os.makedirs("exports/gifs", exist_ok=True)
    gif_path = f"exports/gifs/{filename_label}.gif"

    frame_duration_ms = max(20, int(round(duration_per_move * 1000)))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Highlight GIF exported successfully: {gif_path}")
    return gif_path


def generate_game_webm(
    move_stack,
    filename_label,
    duration_per_move=1.0,
    metadata=None,
    crf=34,
):
    """Compile a game move history into a blog-friendly compressed WebM."""
    frames = _render_game_frames(move_stack, metadata)
    os.makedirs("exports/webm", exist_ok=True)
    webm_path = f"exports/webm/{filename_label}.webm"
    frame_rate = 1 / max(duration_per_move, 0.02)

    with tempfile.TemporaryDirectory(prefix="transformer_chess_webm_") as temp_dir:
        temp_path = Path(temp_dir)
        for index, frame in enumerate(frames):
            frame.save(temp_path / f"frame_{index:05d}.png")

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{frame_rate:.6f}",
            "-i",
            str(temp_path / "frame_%05d.png"),
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "0",
            "-crf",
            str(crf),
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-row-mt",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-an",
            webm_path,
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to export WebM files.") from exc

    print(f"Compressed WebM exported successfully: {webm_path}")
    return webm_path


def generate_game_mp4(
    move_stack,
    filename_label,
    duration_per_move=1.0,
    metadata=None,
    crf=32,
):
    """Compile a game move history into a highly compressed H.264 MP4."""
    frames = _render_game_frames(move_stack, metadata)
    os.makedirs("exports/mp4", exist_ok=True)
    mp4_path = f"exports/mp4/{filename_label}.mp4"
    frame_rate = 1 / max(duration_per_move, 0.02)

    with tempfile.TemporaryDirectory(prefix="transformer_chess_mp4_") as temp_dir:
        temp_path = Path(temp_dir)
        for index, frame in enumerate(frames):
            frame.save(temp_path / f"frame_{index:05d}.png")

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{frame_rate:.6f}",
            "-i",
            str(temp_path / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            mp4_path,
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to export MP4 files.") from exc

    print(f"Compressed MP4 exported successfully: {mp4_path}")
    return mp4_path
