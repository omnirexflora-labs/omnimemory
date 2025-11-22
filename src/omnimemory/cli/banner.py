"""
Dynamic banner generator that measures ASCII art and creates properly fitted box
"""

import unicodedata


def create_omnimemory_banner():
    """Generate OMNIMEMORY banner with properly measured box"""

    ascii_lines = [
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

    def display_width(text):
        """Calculate actual terminal display width"""
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

    def create_centered_line(text, style=""):
        """Create a centered line with proper padding"""
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

if __name__ == "__main__":
    try:
        from rich.console import Console

        console = Console()
        console.print(OMNIMEMORY_BANNER)
        print("\n" + "=" * 50)
        print("Copy the OMNIMEMORY_BANNER variable for use in your code")
    except ImportError:
        print(OMNIMEMORY_BANNER)
