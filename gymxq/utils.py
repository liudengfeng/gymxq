import os
import sys

# from importlib.resources import files
from contextlib import closing
from io import StringIO
from typing import List, Optional

# import pygame
import six
import termcolor
import cppxq

from .constants import (
    BLACK_PLAYER,
    BOARD_NUM_PROMPT,
    NUM_COL,
    NUM_ROW,
    PIECE_HEIGHT,
    PIECE_WIDTH,
    RECORD_NOTES,
    RED_PLAYER,
)

PNG_MAPS = {1: "P", 2: "A", 3: "B", 4: "C", 5: "N", 6: "R", 7: "K"}
COLOR2NUM = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def get_piece_png_file(color_id: int, piece_id: int):
    assert color_id in (RED_PLAYER, BLACK_PLAYER), "颜色编号1-2"
    assert piece_id in range(1, 8), "棋子编号1-7"
    file_name_fmt = "{}{}.png"
    if color_id == RED_PLAYER:
        return file_name_fmt.format("r", PNG_MAPS[piece_id])
    elif color_id == BLACK_PLAYER:
        return file_name_fmt.format("b", PNG_MAPS[piece_id])
    raise ValueError("错误")


def get_index(x: int, y: int):
    return (NUM_ROW - y - 1) * NUM_COL + x


def move_to_coordinate(move_str, convert=False):
    """将移动字符串转换为棋盘数组坐标

    Args:
        move_str (str): 代表移动的4位整数字符串

    Returns:
        tuple: x0,y0,x1,y1 4元组
    """
    assert isinstance(move_str, str)
    assert len(move_str) == 4
    x0 = int(move_str[0])
    y0 = int(move_str[1])
    x1 = int(move_str[2])
    y1 = int(move_str[3])
    if convert:
        return x0, NUM_ROW - y0 - 1, x1, NUM_ROW - y1 - 1
    else:
        return x0, y0, x1, y1


def get_center(x, y):
    """给定x轴、y轴单元格数量,返回窗口的中心点坐标

    Args:
        x (int): x单元格
        y (int): y单元格

    Returns:
        tuple: 窗口x,y坐标
    """
    return (
        x * PIECE_WIDTH + PIECE_WIDTH // 2,
        (NUM_ROW - y) * PIECE_HEIGHT - PIECE_HEIGHT // 2,
    )


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    # Import six here so that `utils` has no import-time dependencies.
    # We want this since we use `utils` during our import-time sanity checks
    # that verify that our dependencies (including six) are actually present.
    attr = []
    num = COLOR2NUM[color]
    if highlight:
        num += 10
    attr.append(six.u(str(num)))
    if bold:
        attr.append(six.u("1"))
    attrs = six.u(";").join(attr)
    return six.u("\x1b[%sm%s\x1b[0m") % (attrs, string)


def render_board_to_text(
    board: cppxq.XqBoard, last_move: Optional[str] = None, title: Optional[str] = None
):
    """棋盘文本表达

    Args:
        board (cppxq.XqBoard): 棋盘实例
        last_move (Optional[str], optional): 最后移动字符串. Defaults to None.
        title (Optional[str], optional): 标题. Defaults to None.
    Note:
        0.5 ms 速度更快
    """
    outfile = StringIO()
    # outfile = sys.stdout
    outfile.write("\n")
    if title:
        outfile.write(title.center(17))
        outfile.write("\n")
    pieces = board.piece_cn_names()
    cn_colors = board.piece_cn_colors()
    for row in range(NUM_ROW):
        for col in range(NUM_COL):
            cn_color = cn_colors[row][col]
            highlight = False
            color = "gray"
            if cn_color == "红":
                color = "red"
            elif cn_color == "黑":
                color = "blue"
            if last_move and last_move[0] == col and last_move[1] == row:
                # 标记开始位置
                highlight = True
                color = "yellow"
            elif last_move and last_move[2] == col and last_move[3] == row:
                highlight = True
            pieces[row][col] = colorize(pieces[row][col], color, highlight=highlight)
    for row in range(NUM_ROW):
        pieces[row].insert(0, str(9 - row) + " ")
    outfile.write("\n".join("".join(line) for line in pieces) + "\n")
    outfile.write(BOARD_NUM_PROMPT + "\n")
    if board.is_finished():
        game_result = board.game_result_string()
        outfile.write(game_result + "\n")
    else:
        next_player = "轮到{}走子".format(board.next_player_string())
        outfile.write(next_player + "\n")

    with closing(outfile):
        return outfile.getvalue()


def colorize_v2(string, color, bold=False):
    attrs = []
    if bold:
        attrs.append("bold")
    return termcolor.colored(string, color, attrs=attrs)


def render_board_to_text_v2(
    board: cppxq.XqBoard, last_move: Optional[str] = None, title: Optional[str] = None
):
    """棋盘文本表达

    Args:
        board (cppxq.XqBoard): 棋盘实例
        last_move (Optional[str], optional): 最后移动字符串. Defaults to None.
        title (Optional[str], optional): 标题. Defaults to None.
    Note:
        1 ms 慢
    """
    # outfile = StringIO()
    outfile = sys.stdout
    if title:
        termcolor.cprint(title.center(17), "blue", attrs=["bold"], file=outfile)
        print()
        # termcolor.cprint("\n", file=outfile)
    pieces = board.piece_cn_names()
    cn_colors = board.piece_cn_colors()
    for row in range(NUM_ROW):
        for col in range(NUM_COL):
            cn_color = cn_colors[row][col]
            highlight = False
            color = "grey"
            if cn_color == "红":
                color = "red"
            elif cn_color == "黑":
                color = "blue"
            if last_move and last_move[0] == col and last_move[1] == row:
                # 标记开始位置
                highlight = True
                color = "yellow"
            elif last_move and last_move[2] == col and last_move[3] == row:
                highlight = True
            pieces[row][col] = colorize_v2(pieces[row][col], color)

    for row in range(NUM_ROW):
        pieces[row].insert(0, str(9 - row) + " ")

    # print("\n".join("".join(line) for line in pieces) + "\n")
    print("\n".join("".join(line) for line in pieces))
    print()

    outfile.write(BOARD_NUM_PROMPT)
    print()

    if board.is_finished():
        game_result = board.game_result_string()
        print(game_result + "\n")
    else:
        next_player = board.next_player_string()
        color = "red" if next_player == "红方" else "blue"
        next_player = colorize_v2(next_player, color)
        next_player = "轮到{}走子".format(next_player)
        print(next_player + "\n")
    print()


def move_from_num_tag(piece, x):
    """移动数字标签

    Args:
        piece (Piece): 棋子对象
        x (int): 移动前或移动后x轴坐标

    Raises:
        ValueError: 不得为空

    Returns:
        str: 数字标识
    """
    if piece.color_id == RED_PLAYER:
        return RECORD_NOTES[9 - x - 1][1]
    elif piece.color_id == BLACK_PLAYER:
        return RECORD_NOTES[x][0]
    raise ValueError("不支持空白棋子")


def move_delta_num_tag(piece, delta):
    """移动变动标签

    Args:
        piece (Piece): 棋子对象
        delta (int): 坐标变动数字

    Raises:
        ValueError: 不得为空

    Returns:
        str: 数字标识
    """
    assert delta >= 1 and delta <= 9, "横向或纵向变动最大为9"
    if piece.color_id == RED_PLAYER:
        return RECORD_NOTES[delta - 1][1]
    elif piece.color_id == BLACK_PLAYER:
        return RECORD_NOTES[delta - 1][0]


def front_and_back_tag(color_id: int, ys: List[int], y: int):
    """棋子前后标识

    Args:
        color_id (int): 颜色编码
        ys (List[int]): 棋盘y轴坐标
        y (int): 标识棋子y轴坐标

    Returns:
        str: 中文前后标识
    """
    reverse = False
    if color_id == RED_PLAYER:
        reverse = True
    ys_sorted = sorted(ys, reverse=reverse)
    if len(ys) == 2:
        ds = ["前", "后"]
        return ds[ys_sorted.index(y)]
    elif len(ys) == 3:
        ds = ["前", "中", "后"]
        return ds[ys_sorted.index(y)]
    else:
        # 注意此处反取，以此表示三连以上
        cid = 0 if color_id == RED_PLAYER else 1
        return "前{}".format(RECORD_NOTES[ys_sorted.index(y)][cid])


def _get_in_same_col_ys(board, to_check, x0):
    # 棋盘上列相同的同类本方棋子y坐标列表
    pieces = board.get_pieces()
    sames = []
    for p in pieces:
        if (
            to_check.color_id == p.color_id
            and p.x == x0
            and to_check.piece_id == p.piece_id
        ):
            sames.append(p.y)
    if to_check.y not in sames:
        sames.append(to_check.y)
    return sames


def _forward_and_backward_tag(piece, y0):
    if piece.color_id == RED_PLAYER:
        if piece.y > y0:
            return "进"
        elif piece.y < y0:
            return "退"
    elif piece.color_id == BLACK_PLAYER:
        if piece.y < y0:
            return "进"
        elif piece.y > y0:
            return "退"


def make_last_move_qipu(board: cppxq.XqBoard, move: str):
    record = ""
    x0, y0, x1, y1 = move_to_coordinate(move)
    piece = board.get_piece(x1, y1)
    assert piece.color_name in ("红", "黑"), "棋子不得为空"
    sames = _get_in_same_col_ys(board, piece, x0)
    # 不存在重叠时或帅、士、象、马棋子移动时
    if len(sames) <= 1 or piece.piece_id in (2, 3, 5, 7):
        # 水平移动
        if y0 == y1:
            record = "{}{}平{}".format(
                piece.piece_name,
                move_from_num_tag(piece, x0),
                move_from_num_tag(piece, x1),
            )
        # 垂直移动：帅
        elif x0 == x1:
            record = "{}{}{}{}".format(
                piece.piece_name,
                move_from_num_tag(piece, x0),
                _forward_and_backward_tag(piece, y0),
                move_delta_num_tag(piece, abs(y0 - y1)),
            )
        # 其他移动：士、象、马
        else:
            record = "{}{}{}{}".format(
                piece.piece_name,
                move_from_num_tag(piece, x0),
                _forward_and_backward_tag(piece, y0),
                move_from_num_tag(piece, x1),
            )
    # 兵、车、炮需要检查前后问题
    # 四兵或以上垂直【实际极少出现此类情形，但不能排除】
    elif len(sames) in (2, 3, 4, 5):
        fb = front_and_back_tag(piece.color_id, sames, piece.y)
        if len(sames) <= 3:
            # 水平移动
            if y0 == y1:
                # 兵
                if piece.piece_id == 1:
                    record = "{}{}平{}".format(
                        fb,
                        # 如前兵三进一，省略后就是前三进一了
                        move_from_num_tag(piece, x0),
                        move_from_num_tag(piece, x1),
                    )
                # 车、炮
                else:
                    record = "{}{}平{}".format(
                        fb,
                        piece.piece_name,
                        move_from_num_tag(piece, x1),
                    )
            # 垂直移动
            elif x0 == x1:
                # 兵
                if piece.piece_id == 1:
                    record = "{}{}{}{}".format(
                        fb,
                        move_from_num_tag(piece, x0),
                        _forward_and_backward_tag(piece, y0),
                        move_delta_num_tag(piece, abs(y0 - y1)),
                    )
                else:
                    record = "{}{}{}{}".format(
                        fb,
                        piece.piece_name,
                        _forward_and_backward_tag(piece, y0),
                        move_delta_num_tag(piece, abs(y0 - y1)),
                    )
        else:
            # 3兵以上
            # 水平移动
            if y0 == y1:
                record = "{}平{}".format(
                    fb,
                    move_from_num_tag(piece, x1),
                )
            # 垂直移动
            elif x0 == x1:
                record = "{}{}{}".format(
                    fb,
                    _forward_and_backward_tag(piece, y0),
                    move_delta_num_tag(piece, abs(y0 - y1)),
                )
    return record
