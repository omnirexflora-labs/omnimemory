"""
Dynamic banner generator that measures ASCII art and creates properly fitted box
"""

from typing import Iterable
import unicodedata


def create_omnimemory_banner() -> str:
    """
    Generate OMNIMEMORY banner with properly measured box.

    Returns:
        Rich-formatted string representing the centered OmniMemory banner.
    """

    ascii_lines: Iterable[str] = [
        "  ██████╗ ███╗   ███╗███╗   ██╗██╗███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗",
        " ██╔═══██╗████╗ ████║████╗  ██║██║████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝",
        " ██║   ██║██╔████╔██║██╔██╗ ██║██║██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝ ",
        " ██║   ██║██║╚██╔╝██║██║╚██╗██║██║██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝  ",
        " ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║   ",
        "  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ",
    ]

    subtitle = "Self-Evolving Composite Memory Synthesis Architecture (SECMSA)"
    description = "Advanced Memory Management System for Autonomous AI Agents"
    features = "Dual-Agent Construction • Asynchronous Processing • Self-Evolving Conflict Resolution"

    def display_width(text: str) -> int:
        """
        Calculate Unicode-aware terminal display width.

        Args:
            text: Candidate string whose rendered width is needed.

        Returns:
            Width in monospace terminal cells.
        """
        width = 0
        for char in text:
            if unicodedata.east_asian_width(char) in ("F", "W"):
                width += 2
            elif 0x2500 <= ord(char) <= 0x259F:
                width += 1
            elif ord(char) >= 0x1F300:
                width += 2
            else:
                width += 1
        return width

    max_width = max(display_width(line) for line in ascii_lines)
    max_width = max(
        max_width,
        display_width(subtitle),
        display_width(description),
        display_width(features),
    )

    padding = 4
    box_inner_width = max_width + (padding * 2)

    top_border = "╔" + "═" * box_inner_width + "╗"
    bottom_border = "╚" + "═" * box_inner_width + "╝"
    empty_line = "║" + " " * box_inner_width + "║"

    def create_centered_line(text: str, style: str = "") -> str:
        """
        Create a centered line with proper padding.

        Args:
            text: Raw text to center within the banner.
            style: Optional Rich markup style tag.

        Returns:
            Rich markup string representing the centered line.
        """
        text_width = display_width(text)
        total_padding = box_inner_width - text_width
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad

        if style:
            return f"║{' ' * left_pad}[{style}]{text}[/]{' ' * right_pad}║"
        else:
            return f"║{' ' * left_pad}{text}{' ' * right_pad}║"

    banner_lines = [f"[bold cyan]{top_border}[/]", f"[bold cyan]{empty_line}[/]"]

    for i, line in enumerate(ascii_lines):
        if i < 2:
            style = "bold bright_cyan"
        elif i < 4:
            style = "bold white"
        else:
            style = "bold bright_cyan"
        banner_lines.append(create_centered_line(line, style))

    banner_lines.extend(
        [
            f"[bold cyan]{empty_line}[/]",
            create_centered_line(subtitle, "bold bright_magenta"),
            create_centered_line(description, "bold cyan"),
            create_centered_line(features, "dim bright_white"),
            f"[bold cyan]{empty_line}[/]",
            f"[bold cyan]{bottom_border}[/]",
        ]
    )

    return "\n".join(banner_lines)


OMNIMEMORY_BANNER = create_omnimemory_banner()
