"""Color generation utilities with theme integration."""

from typing import List, Tuple, Dict
import colorsys
import math
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from .theme_constants import ColorTheme
from .theme_manager import ThemeManager


class ColorGenerator:
    """Generates colors for annotation classes matched to UI themes."""
    
    # Color palettes matched to themes
    PALETTES = {
        ColorTheme.MIDNIGHT_BLUE: [
            # Colorblind-safe palette with blue theme
            "#4a9eff",  # Bright Blue
            "#00d4aa",  # Teal
            "#ffa726",  # Orange
            "#66bb6a",  # Green
            "#ab47bc",  # Purple
            "#ef5350",  # Red
            "#42a5f5",  # Light Blue
            "#26a69a",  # Dark Teal
            "#ffca28",  # Yellow
            "#5c6bc0",  # Indigo
            "#ff7043",  # Deep Orange
            "#8d6e63",  # Brown
            "#78909c",  # Blue Grey
            "#ec407a",  # Pink
            "#9ccc65",  # Light Green
            "#7e57c2",  # Deep Purple
        ],
        ColorTheme.FOREST_DARK: [
            # Nature/forest colors
            "#4caf50",  # Green
            "#8bc34a",  # Light Green
            "#388e3c",  # Dark Green
            "#cddc39",  # Lime
            "#795548",  # Brown
            "#ff9800",  # Orange
            "#607d8b",  # Blue Grey
            "#009688",  # Teal
            "#ffc107",  # Amber
            "#3f51b5",  # Indigo
            "#00bcd4",  # Cyan
            "#ff5722",  # Deep Orange
            "#9c27b0",  # Purple
            "#f44336",  # Red
            "#e91e63",  # Pink
            "#2196f3",  # Blue
        ],
        ColorTheme.VOLCANIC_ASH: [
            # Warm volcanic colors
            "#ff6b35",  # Orange Red
            "#ffa726",  # Orange
            "#ff7043",  # Deep Orange
            "#ffca28",  # Amber
            "#d84315",  # Dark Orange
            "#bf360c",  # Dark Red
            "#6d4c41",  # Brown
            "#78909c",  # Blue Grey
            "#546e7a",  # Dark Blue Grey
            "#37474f",  # Darker Blue Grey
            "#fdd835",  # Yellow
            "#827717",  # Dark Yellow
            "#f57c00",  # Orange
            "#e65100",  # Dark Orange
            "#ff3d00",  # Red Orange
            "#dd2c00",  # Dark Red Orange
        ],
        ColorTheme.DEEP_OCEAN: [
            # Ocean/water colors
            "#00acc1",  # Cyan
            "#26c6da",  # Light Cyan
            "#00838f",  # Dark Cyan
            "#00bcd4",  # Cyan
            "#0097a7",  # Dark Cyan
            "#006064",  # Very Dark Cyan
            "#4dd0e1",  # Light Cyan
            "#18ffff",  # Bright Cyan
            "#00e5ff",  # Light Blue
            "#00b8d4",  # Cyan Blue
            "#0091ea",  # Blue
            "#0277bd",  # Dark Blue
            "#01579b",  # Very Dark Blue
            "#80deea",  # Very Light Cyan
            "#b2ebf2",  # Pale Cyan
            "#e0f7fa",  # Very Pale Cyan
        ],
        ColorTheme.CYBERPUNK: [
            # Neon cyberpunk colors
            "#ff0080",  # Hot Pink
            "#00ffff",  # Cyan
            "#ff00ff",  # Magenta
            "#00ff00",  # Green
            "#ffff00",  # Yellow
            "#ff0040",  # Red
            "#00ff80",  # Spring Green
            "#8000ff",  # Purple
            "#ff8000",  # Orange
            "#0080ff",  # Sky Blue
            "#ff0000",  # Pure Red
            "#00b0ff",  # Light Blue
            "#b000ff",  # Violet
            "#ff00b0",  # Pink Purple
            "#40ff00",  # Lime
            "#ff4000",  # Red Orange
        ],
        ColorTheme.SOFT_PASTEL: [
            # Soft pastel colors
            "#f48fb1",  # Pink
            "#90caf9",  # Blue
            "#a5d6a7",  # Green
            "#ffcc80",  # Orange
            "#ce93d8",  # Purple
            "#80deea",  # Cyan
            "#ffab91",  # Deep Orange
            "#81c784",  # Light Green
            "#e1bee7",  # Light Purple
            "#b39ddb",  # Deep Purple
            "#9fa8da",  # Indigo
            "#c5cae9",  # Light Indigo
            "#bbdefb",  # Light Blue
            "#b3e5fc",  # Light Cyan
            "#b2dfdb",  # Teal
            "#c8e6c9",  # Light Green
        ],
        ColorTheme.NORDIC_LIGHT: [
            # Nordic/Scandinavian colors
            "#5e81ac",  # Blue
            "#88c0d0",  # Light Blue
            "#81a1c1",  # Blue Grey
            "#b48ead",  # Purple
            "#a3be8c",  # Green
            "#ebcb8b",  # Yellow
            "#d08770",  # Orange
            "#bf616a",  # Red
            "#4c6a92",  # Dark Blue
            "#6ba3b8",  # Light Blue
            "#6d8bad",  # Blue Grey
            "#9c7b9c",  # Purple
            "#8ca67a",  # Green
            "#d4b57a",  # Yellow
            "#b87760",  # Orange
            "#a5545c",  # Red
        ],
        ColorTheme.WARM_EARTH: [
            # Warm earth tones
            "#8d6e63",  # Brown
            "#a1887f",  # Light Brown
            "#6d4c41",  # Dark Brown
            "#bcaaa4",  # Light Brown Grey
            "#8c7b75",  # Brown Grey
            "#5d4037",  # Dark Brown
            "#d7ccc8",  # Very Light Brown
            "#efebe9",  # Pale Brown
            "#3e2723",  # Very Dark Brown
            "#4e342e",  # Dark Brown
            "#795548",  # Medium Brown
            "#a1887f",  # Light Brown
            "#d7ccc8",  # Very Light Brown
            "#ffab91",  # Light Orange
            "#ff8a65",  # Orange
            "#ff7043",  # Deep Orange
        ],
        ColorTheme.SKY_BLUE: [
            # Sky and cloud colors
            "#2196f3",  # Blue
            "#03dac6",  # Teal
            "#42a5f5",  # Light Blue
            "#1e88e5",  # Medium Blue
            "#1976d2",  # Dark Blue
            "#1565c0",  # Darker Blue
            "#0d47a1",  # Very Dark Blue
            "#64b5f6",  # Light Blue
            "#90caf9",  # Lighter Blue
            "#bbdefb",  # Very Light Blue
            "#e3f2fd",  # Pale Blue
            "#00b8a3",  # Teal
            "#00a693",  # Dark Teal
            "#009485",  # Darker Teal
            "#00bcd4",  # Cyan
            "#00acc1",  # Dark Cyan
        ],
        ColorTheme.SPRING_MEADOW: [
            # Spring/nature colors
            "#66bb6a",  # Green
            "#ffd54f",  # Yellow
            "#4caf50",  # Medium Green
            "#ffb300",  # Amber
            "#43a047",  # Dark Green
            "#388e3c",  # Darker Green
            "#2e7d32",  # Very Dark Green
            "#81c784",  # Light Green
            "#a5d6a7",  # Lighter Green
            "#c8e6c9",  # Very Light Green
            "#e8f5e9",  # Pale Green
            "#fff176",  # Light Yellow
            "#fff59d",  # Lighter Yellow
            "#fff9c4",  # Very Light Yellow
            "#fffde7",  # Pale Yellow
            "#f9a825",  # Dark Yellow
        ]
    }
    
    # Graph colors optimized for each theme's background
    GRAPH_COLORS = {
        ColorTheme.MIDNIGHT_BLUE: [
            "#4a9eff",  # Bright Blue
            "#00d4aa",  # Teal
            "#ffa726",  # Orange
            "#66bb6a",  # Green
            "#ab47bc",  # Purple
            "#ef5350",  # Red
        ],
        ColorTheme.FOREST_DARK: [
            "#66bb6a",  # Green
            "#ffa726",  # Orange
            "#42a5f5",  # Blue
            "#ab47bc",  # Purple
            "#ffca28",  # Yellow
            "#26a69a",  # Teal
        ],
        ColorTheme.VOLCANIC_ASH: [
            "#ff7043",  # Deep Orange
            "#ffca28",  # Amber
            "#ffa726",  # Orange
            "#78909c",  # Blue Grey
            "#8d6e63",  # Brown
            "#ef5350",  # Red
        ],
        ColorTheme.DEEP_OCEAN: [
            "#26c6da",  # Light Cyan
            "#42a5f5",  # Light Blue
            "#00acc1",  # Cyan
            "#5c6bc0",  # Indigo
            "#26a69a",  # Teal
            "#66bb6a",  # Green
        ],
        ColorTheme.CYBERPUNK: [
            "#00ffff",  # Cyan
            "#ff0080",  # Hot Pink
            "#00ff00",  # Green
            "#ffff00",  # Yellow
            "#ff00ff",  # Magenta
            "#0080ff",  # Sky Blue
        ],
        ColorTheme.SOFT_PASTEL: [
            "#5c6bc0",  # Darker Indigo (visible on light)
            "#ec407a",  # Darker Pink
            "#66bb6a",  # Darker Green
            "#ff8a65",  # Darker Orange
            "#ab47bc",  # Darker Purple
            "#26a69a",  # Darker Teal
        ],
        ColorTheme.NORDIC_LIGHT: [
            "#5e81ac",  # Blue
            "#a3be8c",  # Green
            "#ebcb8b",  # Yellow
            "#d08770",  # Orange
            "#b48ead",  # Purple
            "#88c0d0",  # Light Blue
        ],
        ColorTheme.WARM_EARTH: [
            "#6d4c41",  # Dark Brown
            "#ff7043",  # Deep Orange
            "#8d6e63",  # Brown
            "#795548",  # Medium Brown
            "#d84315",  # Dark Orange
            "#5d4037",  # Darker Brown
        ],
        ColorTheme.SKY_BLUE: [
            "#1976d2",  # Dark Blue
            "#00897b",  # Dark Teal
            "#43a047",  # Dark Green
            "#f57c00",  # Dark Orange
            "#8e24aa",  # Dark Purple
            "#d32f2f",  # Dark Red
        ],
        ColorTheme.SPRING_MEADOW: [
            "#388e3c",  # Dark Green
            "#f9a825",  # Dark Yellow
            "#43a047",  # Medium Dark Green
            "#ef6c00",  # Dark Orange
            "#00897b",  # Dark Teal
            "#c62828",  # Dark Red
        ]
    }
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def generate_colors(num_classes: int, theme: str = ColorTheme.MIDNIGHT_BLUE) -> List[Tuple[int, int, int]]:
        """Generate colors for the given number of classes based on theme.
        
        Args:
            num_classes: Number of classes to generate colors for
            theme: Color theme to use
            
        Returns:
            List of RGB tuples
        """
        colors = []
        
        # Get the theme's palette
        base_palette = ColorGenerator.PALETTES.get(theme, ColorGenerator.PALETTES[ColorTheme.MIDNIGHT_BLUE])
        
        # Use predefined palette colors first
        for i, hex_color in enumerate(base_palette):
            if i >= num_classes:
                break
            colors.append(ColorGenerator.hex_to_rgb(hex_color))
        
        # If we need more colors, generate them algorithmically
        if num_classes > len(base_palette):
            colors.extend(ColorGenerator._generate_additional_colors(
                num_classes - len(base_palette),
                len(base_palette),
                theme
            ))
        
        return colors[:num_classes]
    
    @staticmethod
    def get_graph_colors(theme: str = ColorTheme.MIDNIGHT_BLUE) -> List[str]:
        """Get graph colors optimized for the theme's background.
        
        Args:
            theme: Color theme to use
            
        Returns:
            List of hex color strings
        """
        return ColorGenerator.GRAPH_COLORS.get(theme, ColorGenerator.GRAPH_COLORS[ColorTheme.MIDNIGHT_BLUE])
    
    @staticmethod
    def _generate_additional_colors(count: int, start_offset: int, theme: str) -> List[Tuple[int, int, int]]:
        """Generate additional distinguishable colors using HSV color space.
        
        Args:
            count: Number of colors to generate
            start_offset: Offset to avoid collision with predefined colors
            theme: Color theme being used
            
        Returns:
            List of RGB tuples
        """
        colors = []
        
        # Use golden ratio for hue distribution
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Get theme colors to determine if it's a dark theme
        theme_manager = ThemeManager()
        theme_colors = theme_manager.get_theme_colors(theme)
        is_dark = theme_colors.is_dark
        
        # Adjust generation parameters based on theme
        if theme == ColorTheme.CYBERPUNK:
            # High saturation for neon effect
            sat_range = (0.9, 1.0)
            val_range = (0.9, 1.0)
        elif theme in [ColorTheme.SOFT_PASTEL, ColorTheme.NORDIC_LIGHT, ColorTheme.SKY_BLUE]:
            # Lower saturation for softer colors
            sat_range = (0.3, 0.5)
            val_range = (0.7, 0.9) if not is_dark else (0.8, 1.0)
        elif theme in [ColorTheme.WARM_EARTH, ColorTheme.SPRING_MEADOW]:
            # Moderate saturation for natural colors
            sat_range = (0.5, 0.7)
            val_range = (0.5, 0.7) if not is_dark else (0.7, 0.9)
        else:
            # Default for other themes
            sat_range = (0.6, 0.8)
            val_range = (0.7, 0.9) if is_dark else (0.5, 0.7)
        
        # Generate colors with varying hue, saturation, and value
        for i in range(count):
            # Distribute hues using golden ratio
            hue = ((start_offset + i) * golden_ratio) % 1.0
            
            # Vary saturation and value to increase distinguishability
            saturation = sat_range[0] + (sat_range[1] - sat_range[0]) * (i % 3) / 2
            value = val_range[0] + (val_range[1] - val_range[0]) * ((i + 1) % 3) / 2
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
        return colors
    
    @staticmethod
    def get_pen_style(class_id: int, total_classes: int) -> Qt.PenStyle:
        """Get pen style for a class to add additional distinction.
        
        Uses dashed patterns for classes that might have similar colors.
        
        Args:
            class_id: The class ID
            total_classes: Total number of classes
            
        Returns:
            Qt.PenStyle for the class
        """
        if total_classes <= 8:
            # For small number of classes, use solid lines
            return Qt.PenStyle.SolidLine
        
        # For larger number of classes, use patterns to distinguish
        # Classes are grouped in sets of 8 with different patterns
        pattern_group = class_id // 8
        
        patterns = [
            Qt.PenStyle.SolidLine,
            Qt.PenStyle.DashLine,
            Qt.PenStyle.DotLine,
            Qt.PenStyle.DashDotLine,
            Qt.PenStyle.DashDotDotLine,
        ]
        
        return patterns[pattern_group % len(patterns)]
    
    @staticmethod
    def get_contrast_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get a contrasting color for text on the given background.
        
        Args:
            rgb: Background color as RGB tuple
            
        Returns:
            RGB tuple for contrasting text color
        """
        # Calculate relative luminance
        r, g, b = [x / 255.0 for x in rgb]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Return white or black based on luminance
        return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)
    
    @staticmethod
    def adjust_for_dark_theme(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Adjust color to ensure visibility on dark background.
        
        Args:
            rgb: Original RGB color
            
        Returns:
            Adjusted RGB color
        """
        # Convert to HSV
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Ensure minimum brightness for dark backgrounds
        v = max(v, 0.4)
        
        # Ensure moderate saturation
        s = min(s, 0.9)
        
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))


class AnnotationColorManager:
    """Manages annotation colors with theme integration."""
    
    def __init__(self, num_classes: int = 80, theme: str = ColorTheme.MIDNIGHT_BLUE):
        """Initialize color manager.
        
        Args:
            num_classes: Maximum number of classes to support
            theme: Color theme to use
        """
        self.num_classes = num_classes
        self._theme = theme
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}
        self._theme_manager = ThemeManager()
        self._theme_manager.set_theme(theme)
        self._generate_colors()
    
    def set_theme(self, theme: str):
        """Change the color theme.
        
        Args:
            theme: New color theme to use
        """
        if theme != self._theme:
            self._theme = theme
            self._theme_manager.set_theme(theme)
            self._color_cache.clear()
            self._generate_colors()
    
    def get_theme(self) -> str:
        """Get the current theme name."""
        return self._theme
    
    def get_theme_manager(self) -> ThemeManager:
        """Get the theme manager instance."""
        return self._theme_manager
    
    def _generate_colors(self):
        """Generate and cache colors for all classes."""
        colors = ColorGenerator.generate_colors(self.num_classes, self._theme)
        theme_colors = self._theme_manager.get_theme_colors(self._theme)
        
        for i, color in enumerate(colors):
            # Adjust for theme background if needed
            if theme_colors.is_dark:
                self._color_cache[i] = ColorGenerator.adjust_for_dark_theme(color)
            else:
                # For light themes, ensure colors are dark enough
                r, g, b = color
                # Darken if too light
                if (r + g + b) / 3 > 200:
                    self._color_cache[i] = (int(r * 0.7), int(g * 0.7), int(b * 0.7))
                else:
                    self._color_cache[i] = color
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID.
        
        Args:
            class_id: The class ID
            
        Returns:
            RGB color tuple
        """
        # Use modulo for classes beyond our generated range
        effective_id = class_id % self.num_classes
        return self._color_cache.get(effective_id, (255, 255, 255))
    
    def get_qcolor(self, class_id: int) -> QColor:
        """Get QColor for a class ID.
        
        Args:
            class_id: The class ID
            
        Returns:
            QColor object
        """
        rgb = self.get_color(class_id)
        return QColor(*rgb)
    
    def get_pen_style(self, class_id: int) -> Qt.PenStyle:
        """Get pen style for a class ID.
        
        Args:
            class_id: The class ID
            
        Returns:
            Qt.PenStyle
        """
        return ColorGenerator.get_pen_style(class_id, self.num_classes)
    
    def get_text_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get contrasting text color for labels on class background.
        
        Args:
            class_id: The class ID
            
        Returns:
            RGB color tuple for text
        """
        bg_color = self.get_color(class_id)
        return ColorGenerator.get_contrast_color(bg_color)
    
    def get_graph_colors(self) -> List[str]:
        """Get colors optimized for graphs on current theme.
        
        Returns:
            List of hex color strings
        """
        return ColorGenerator.get_graph_colors(self._theme)