"""Generate Koe icon - sound bars style."""
from PIL import Image, ImageDraw
from pathlib import Path

def create_icon(size=256):
    """Create a sound bars icon - fills entire canvas edge-to-edge."""
    bar_color = (0, 255, 136, 255)  # Terminal green #00ff88

    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent
    draw = ImageDraw.Draw(img)

    # Three bars filling the full width with minimal gaps
    gap = size // 24  # Small gap between bars
    bar_width = (size - 2 * gap) // 3

    # Bar heights - fill full height
    bar_heights = [0.6, 1.0, 0.8]  # Left, middle, right
    bar_bottom = size  # Bars go to very bottom

    for i, height_frac in enumerate(bar_heights):
        x = i * (bar_width + gap)
        bar_height = int(size * height_frac)
        y_top = bar_bottom - bar_height

        bar_radius = bar_width // 5

        draw.rounded_rectangle(
            [x, y_top, x + bar_width, bar_bottom],
            radius=bar_radius,
            fill=bar_color
        )

    return img

def main():
    # Get the Koe root directory (parent of scripts/)
    script_dir = Path(__file__).parent
    koe_root = script_dir.parent
    assets_dir = koe_root / "assets"

    # Create main PNG icon
    icon_256 = create_icon(256)
    png_path = assets_dir / "koe-icon.png"
    icon_256.save(png_path, 'PNG')
    print(f"Created {png_path}")

    # Create ICO with multiple sizes - largest first
    sizes = [256, 48, 32, 24, 16]
    icons = [create_icon(s) for s in sizes]

    # Save as ICO - use the 256 as base, append smaller sizes
    ico_path = assets_dir / "koe-icon.ico"
    icons[0].save(
        ico_path,
        format='ICO',
        append_images=icons[1:]
    )
    print(f"Created {ico_path}")

    # Also create a simple tray-optimized version (higher contrast)
    tray_icon = create_tray_icon(32)
    tray_path = assets_dir / "koe-tray.png"
    tray_icon.save(tray_path, 'PNG')
    print(f"Created {tray_path}")

def create_tray_icon(size=32):
    """Create a high-contrast tray icon - fills entire canvas."""
    bar_color = (0, 255, 136, 255)  # Terminal green

    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent
    draw = ImageDraw.Draw(img)

    # Three bars filling full width
    gap = max(1, size // 24)
    bar_width = (size - 2 * gap) // 3

    bar_heights = [0.6, 1.0, 0.8]
    bar_bottom = size

    for i, h_frac in enumerate(bar_heights):
        x = i * (bar_width + gap)
        bar_h = int(size * h_frac)
        y_top = bar_bottom - bar_h

        draw.rectangle(
            [x, y_top, x + bar_width - 1, bar_bottom],
            fill=bar_color
        )

    return img

if __name__ == '__main__':
    main()
