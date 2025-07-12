"""Theme management system with comprehensive color schemes."""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PyQt6.QtCore import QObject, pyqtSignal
from .theme_constants import ColorTheme


@dataclass
class ThemeColors:
    """Colors for a complete theme."""
    # UI Colors
    background: str
    surface: str
    surface_variant: str
    primary: str
    primary_variant: str
    secondary: str
    secondary_variant: str
    text_primary: str
    text_secondary: str
    text_disabled: str
    border: str
    border_hover: str
    error: str
    success: str
    warning: str
    info: str
    
    # Special UI elements
    scrollbar_bg: str
    scrollbar_handle: str
    scrollbar_handle_hover: str
    selection_bg: str
    selection_text: str
    tooltip_bg: str
    tooltip_text: str
    
    # Chart colors
    chart_bg: str
    chart_grid: str
    chart_text: str
    
    # Is this a dark theme?
    is_dark: bool


class ThemeManager(QObject):
    """Manages application themes and color schemes."""
    
    themeChanged = pyqtSignal(str)  # Emitted when theme changes
    
    # Theme definitions
    THEMES = {
        ColorTheme.MIDNIGHT_BLUE: ThemeColors(
            # Dark theme with blue accents
            background="#0a0e1a",
            surface="#151923",
            surface_variant="#1e2330",
            primary="#4a9eff",
            primary_variant="#2e7cd6",
            secondary="#00d4aa",
            secondary_variant="#00a383",
            text_primary="#e8eaed",
            text_secondary="#a8b2c7",
            text_disabled="#5f6b7c",
            border="#2d3442",
            border_hover="#3d4452",
            error="#ff5252",
            success="#00e676",
            warning="#ffd600",
            info="#40c4ff",
            scrollbar_bg="#1a1f2e",
            scrollbar_handle="#3d4452",
            scrollbar_handle_hover="#4d5462",
            selection_bg="#2e7cd6",
            selection_text="#ffffff",
            tooltip_bg="#2d3442",
            tooltip_text="#e8eaed",
            chart_bg="#0f1419",
            chart_grid="#2d3442",
            chart_text="#a8b2c7",
            is_dark=True
        ),
        
        ColorTheme.FOREST_DARK: ThemeColors(
            # Dark theme with green nature colors
            background="#0d1611",
            surface="#1a2620",
            surface_variant="#243028",
            primary="#4caf50",
            primary_variant="#388e3c",
            secondary="#8bc34a",
            secondary_variant="#689f38",
            text_primary="#e0f2e0",
            text_secondary="#a3c9a3",
            text_disabled="#648964",
            border="#2e3e32",
            border_hover="#3e4e42",
            error="#f44336",
            success="#00e676",
            warning="#ff9800",
            info="#03a9f4",
            scrollbar_bg="#1a2620",
            scrollbar_handle="#3e4e42",
            scrollbar_handle_hover="#4e5e52",
            selection_bg="#388e3c",
            selection_text="#ffffff",
            tooltip_bg="#2e3e32",
            tooltip_text="#e0f2e0",
            chart_bg="#111a15",
            chart_grid="#2e3e32",
            chart_text="#a3c9a3",
            is_dark=True
        ),
        
        ColorTheme.VOLCANIC_ASH: ThemeColors(
            # Dark theme with orange/red volcanic colors
            background="#1a0f0a",
            surface="#2d1810",
            surface_variant="#3d2418",
            primary="#ff6b35",
            primary_variant="#e55100",
            secondary="#ffa726",
            secondary_variant="#fb8c00",
            text_primary="#ffe0d0",
            text_secondary="#dba590",
            text_disabled="#8b6550",
            border="#4d3020",
            border_hover="#5d4030",
            error="#ff1744",
            success="#00e676",
            warning="#ffea00",
            info="#00b0ff",
            scrollbar_bg="#2d1810",
            scrollbar_handle="#5d4030",
            scrollbar_handle_hover="#6d5040",
            selection_bg="#e55100",
            selection_text="#ffffff",
            tooltip_bg="#4d3020",
            tooltip_text="#ffe0d0",
            chart_bg="#1f1208",
            chart_grid="#4d3020",
            chart_text="#dba590",
            is_dark=True
        ),
        
        ColorTheme.DEEP_OCEAN: ThemeColors(
            # Dark theme with deep ocean colors
            background="#051e3e",
            surface="#0a2a4e",
            surface_variant="#0f365e",
            primary="#00acc1",
            primary_variant="#008a99",
            secondary="#26c6da",
            secondary_variant="#00b8d4",
            text_primary="#e0f7fa",
            text_secondary="#80deea",
            text_disabled="#4d8a99",
            border="#1a4a6e",
            border_hover="#2a5a7e",
            error="#ff5252",
            success="#69f0ae",
            warning="#ffd740",
            info="#448aff",
            scrollbar_bg="#0a2a4e",
            scrollbar_handle="#2a5a7e",
            scrollbar_handle_hover="#3a6a8e",
            selection_bg="#008a99",
            selection_text="#ffffff",
            tooltip_bg="#1a4a6e",
            tooltip_text="#e0f7fa",
            chart_bg="#03152a",
            chart_grid="#1a4a6e",
            chart_text="#80deea",
            is_dark=True
        ),
        
        ColorTheme.CYBERPUNK: ThemeColors(
            # Dark theme with neon cyberpunk colors
            background="#0a0a0f",
            surface="#16161f",
            surface_variant="#22222f",
            primary="#ff0080",
            primary_variant="#e6006e",
            secondary="#00ffff",
            secondary_variant="#00e5e5",
            text_primary="#ffffff",
            text_secondary="#b0b0ff",
            text_disabled="#6060a0",
            border="#3030ff",
            border_hover="#4040ff",
            error="#ff0040",
            success="#00ff40",
            warning="#ffff00",
            info="#00b0ff",
            scrollbar_bg="#16161f",
            scrollbar_handle="#3030ff",
            scrollbar_handle_hover="#4040ff",
            selection_bg="#ff0080",
            selection_text="#ffffff",
            tooltip_bg="#22222f",
            tooltip_text="#ffffff",
            chart_bg="#0d0d12",
            chart_grid="#3030ff",
            chart_text="#b0b0ff",
            is_dark=True
        ),
        
        ColorTheme.SOFT_PASTEL: ThemeColors(
            # Light theme with soft pastel colors
            background="#fdf8f5",
            surface="#ffffff",
            surface_variant="#f5f0ed",
            primary="#f48fb1",
            primary_variant="#ec407a",
            secondary="#90caf9",
            secondary_variant="#42a5f5",
            text_primary="#3e2723",
            text_secondary="#6d4c41",
            text_disabled="#a1887f",
            border="#e0d5d0",
            border_hover="#d0c5c0",
            error="#e91e63",
            success="#66bb6a",
            warning="#ffa726",
            info="#42a5f5",
            scrollbar_bg="#f5f0ed",
            scrollbar_handle="#e0d5d0",
            scrollbar_handle_hover="#d0c5c0",
            selection_bg="#f48fb1",
            selection_text="#3e2723",
            tooltip_bg="#3e2723",
            tooltip_text="#ffffff",
            chart_bg="#fcf5f2",
            chart_grid="#e0d5d0",
            chart_text="#6d4c41",
            is_dark=False
        ),
        
        ColorTheme.NORDIC_LIGHT: ThemeColors(
            # Light theme with Nordic/Scandinavian colors
            background="#f8f9fb",
            surface="#ffffff",
            surface_variant="#eceff4",
            primary="#5e81ac",
            primary_variant="#4c6a92",
            secondary="#88c0d0",
            secondary_variant="#6ba3b8",
            text_primary="#2e3440",
            text_secondary="#4c566a",
            text_disabled="#a0a8b8",
            border="#d8dee9",
            border_hover="#c8ceda",
            error="#bf616a",
            success="#a3be8c",
            warning="#ebcb8b",
            info="#81a1c1",
            scrollbar_bg="#eceff4",
            scrollbar_handle="#d8dee9",
            scrollbar_handle_hover="#c8ceda",
            selection_bg="#5e81ac",
            selection_text="#ffffff",
            tooltip_bg="#2e3440",
            tooltip_text="#eceff4",
            chart_bg="#f5f6f8",
            chart_grid="#d8dee9",
            chart_text="#4c566a",
            is_dark=False
        ),
        
        ColorTheme.WARM_EARTH: ThemeColors(
            # Light theme with warm earth tones
            background="#faf8f3",
            surface="#ffffff",
            surface_variant="#f5f0e6",
            primary="#8d6e63",
            primary_variant="#6d4c41",
            secondary="#a1887f",
            secondary_variant="#8c7b75",
            text_primary="#3e2723",
            text_secondary="#5d4037",
            text_disabled="#a08070",
            border="#d7ccc8",
            border_hover="#c7bcb8",
            error="#d32f2f",
            success="#689f38",
            warning="#f57c00",
            info="#0288d1",
            scrollbar_bg="#f5f0e6",
            scrollbar_handle="#d7ccc8",
            scrollbar_handle_hover="#c7bcb8",
            selection_bg="#8d6e63",
            selection_text="#ffffff",
            tooltip_bg="#3e2723",
            tooltip_text="#faf8f3",
            chart_bg="#f7f5f0",
            chart_grid="#d7ccc8",
            chart_text="#5d4037",
            is_dark=False
        ),
        
        ColorTheme.SKY_BLUE: ThemeColors(
            # Light theme with sky and cloud colors
            background="#f0f8ff",
            surface="#ffffff",
            surface_variant="#e6f3ff",
            primary="#2196f3",
            primary_variant="#1976d2",
            secondary="#03dac6",
            secondary_variant="#00b8a3",
            text_primary="#0d1b2a",
            text_secondary="#415a77",
            text_disabled="#778da9",
            border="#c5dae8",
            border_hover="#b5cad8",
            error="#f44336",
            success="#4caf50",
            warning="#ff9800",
            info="#00bcd4",
            scrollbar_bg="#e6f3ff",
            scrollbar_handle="#c5dae8",
            scrollbar_handle_hover="#b5cad8",
            selection_bg="#2196f3",
            selection_text="#ffffff",
            tooltip_bg="#0d1b2a",
            tooltip_text="#f0f8ff",
            chart_bg="#f5faff",
            chart_grid="#c5dae8",
            chart_text="#415a77",
            is_dark=False
        ),
        
        ColorTheme.SPRING_MEADOW: ThemeColors(
            # Light theme with spring/nature colors
            background="#f5fdf5",
            surface="#ffffff",
            surface_variant="#e8f5e9",
            primary="#66bb6a",
            primary_variant="#4caf50",
            secondary="#ffd54f",
            secondary_variant="#ffb300",
            text_primary="#1b5e20",
            text_secondary="#2e7d32",
            text_disabled="#81c784",
            border="#c8e6c9",
            border_hover="#b8d6b9",
            error="#e53935",
            success="#43a047",
            warning="#fb8c00",
            info="#039be5",
            scrollbar_bg="#e8f5e9",
            scrollbar_handle="#c8e6c9",
            scrollbar_handle_hover="#b8d6b9",
            selection_bg="#66bb6a",
            selection_text="#ffffff",
            tooltip_bg="#1b5e20",
            tooltip_text="#f5fdf5",
            chart_bg="#f8fef8",
            chart_grid="#c8e6c9",
            chart_text="#2e7d32",
            is_dark=False
        ),
        
        # Additional Dark Themes
        ColorTheme.DARK_AMBER: ThemeColors(
            # Dark theme with amber/gold accents
            background="#0f0e0a",
            surface="#1a1915",
            surface_variant="#252420",
            primary="#d4a017",  # Darker amber for better contrast
            primary_variant="#b8860b",  # Darker gold 
            secondary="#ff6f00",
            secondary_variant="#ff5722",
            text_primary="#fef7e0",
            text_secondary="#e6d5a8",
            text_disabled="#8b7d5c",
            border="#3d3a30",
            border_hover="#4d4a40",
            error="#ff5252",
            success="#00e676",
            warning="#ff9800",
            info="#40c4ff",
            scrollbar_bg="#1a1915",
            scrollbar_handle="#4d4a40",
            scrollbar_handle_hover="#5d5a50",
            selection_bg="#b8860b",  # Darker for better contrast
            selection_text="#000000",
            tooltip_bg="#3d3a30",
            tooltip_text="#fef7e0",
            chart_bg="#121109",
            chart_grid="#3d3a30",
            chart_text="#e6d5a8",
            is_dark=True
        ),
        
        ColorTheme.TWILIGHT_PURPLE: ThemeColors(
            # Dark theme with purple twilight colors
            background="#0e0a1a",
            surface="#191523",
            surface_variant="#231e30",
            primary="#9c27b0",
            primary_variant="#7b1fa2",
            secondary="#e91e63",
            secondary_variant="#c2185b",
            text_primary="#f3e5f5",
            text_secondary="#ce93d8",
            text_disabled="#7e57c2",
            border="#3d2d4a",
            border_hover="#4d3d5a",
            error="#ff5252",
            success="#69f0ae",
            warning="#ffd740",
            info="#448aff",
            scrollbar_bg="#191523",
            scrollbar_handle="#4d3d5a",
            scrollbar_handle_hover="#5d4d6a",
            selection_bg="#7b1fa2",
            selection_text="#ffffff",
            tooltip_bg="#3d2d4a",
            tooltip_text="#f3e5f5",
            chart_bg="#110d19",
            chart_grid="#3d2d4a",
            chart_text="#ce93d8",
            is_dark=True
        ),
        
        ColorTheme.CARBON_GRAY: ThemeColors(
            # Dark monochromatic theme
            background="#0a0a0a",
            surface="#1a1a1a",
            surface_variant="#252525",
            primary="#757575",
            primary_variant="#616161",
            secondary="#9e9e9e",
            secondary_variant="#858585",
            text_primary="#e0e0e0",
            text_secondary="#b0b0b0",
            text_disabled="#606060",
            border="#303030",
            border_hover="#404040",
            error="#ff5252",
            success="#00e676",
            warning="#ffd600",
            info="#40c4ff",
            scrollbar_bg="#1a1a1a",
            scrollbar_handle="#404040",
            scrollbar_handle_hover="#505050",
            selection_bg="#616161",
            selection_text="#ffffff",
            tooltip_bg="#303030",
            tooltip_text="#e0e0e0",
            chart_bg="#0f0f0f",
            chart_grid="#303030",
            chart_text="#b0b0b0",
            is_dark=True
        ),
        
        # Additional Light Themes
        ColorTheme.CREAM_COFFEE: ThemeColors(
            # Light theme with coffee/cream colors
            background="#fef9f3",
            surface="#ffffff",
            surface_variant="#f8f2e6",
            primary="#795548",
            primary_variant="#5d4037",
            secondary="#ff9800",
            secondary_variant="#f57c00",
            text_primary="#3e2723",
            text_secondary="#5d4037",
            text_disabled="#a1887f",
            border="#e0d4c8",
            border_hover="#d0c4b8",
            error="#d32f2f",
            success="#689f38",
            warning="#f57c00",
            info="#0288d1",
            scrollbar_bg="#f8f2e6",
            scrollbar_handle="#e0d4c8",
            scrollbar_handle_hover="#d0c4b8",
            selection_bg="#795548",
            selection_text="#ffffff",
            tooltip_bg="#3e2723",
            tooltip_text="#fef9f3",
            chart_bg="#fcf7f1",
            chart_grid="#e0d4c8",
            chart_text="#5d4037",
            is_dark=False
        ),
        
        ColorTheme.LAVENDER_MIST: ThemeColors(
            # Light theme with soft lavender colors
            background="#f8f6ff",
            surface="#ffffff",
            surface_variant="#f0ecff",
            primary="#7c4dff",
            primary_variant="#651fff",
            secondary="#e040fb",
            secondary_variant="#d500f9",
            text_primary="#311b92",
            text_secondary="#4a148c",
            text_disabled="#9575cd",
            border="#e1d5ff",
            border_hover="#d1c5ef",
            error="#d32f2f",
            success="#689f38",
            warning="#f57c00",
            info="#0288d1",
            scrollbar_bg="#f0ecff",
            scrollbar_handle="#e1d5ff",
            scrollbar_handle_hover="#d1c5ef",
            selection_bg="#7c4dff",
            selection_text="#ffffff",
            tooltip_bg="#311b92",
            tooltip_text="#f8f6ff",
            chart_bg="#f5f3fc",
            chart_grid="#e1d5ff",
            chart_text="#4a148c",
            is_dark=False
        ),
        
        ColorTheme.MINT_FRESH: ThemeColors(
            # Light theme with refreshing mint colors
            background="#f0fdf4",
            surface="#ffffff",
            surface_variant="#e0f7ea",
            primary="#00bfa5",
            primary_variant="#00897b",
            secondary="#00e5ff",
            secondary_variant="#00b8d4",
            text_primary="#004d40",
            text_secondary="#00695c",
            text_disabled="#4db6ac",
            border="#b2dfdb",
            border_hover="#a2cfcb",
            error="#d32f2f",
            success="#00c853",
            warning="#ff6f00",
            info="#0091ea",
            scrollbar_bg="#e0f7ea",
            scrollbar_handle="#b2dfdb",
            scrollbar_handle_hover="#a2cfcb",
            selection_bg="#00bfa5",
            selection_text="#ffffff",
            tooltip_bg="#004d40",
            tooltip_text="#f0fdf4",
            chart_bg="#e8faf5",
            chart_grid="#b2dfdb",
            chart_text="#00695c",
            is_dark=False
        ),
        
        # Accessibility Themes
        ColorTheme.HIGH_CONTRAST_DARK: ThemeColors(
            # Maximum contrast white on black
            background="#000000",
            surface="#000000",
            surface_variant="#1a1a1a",
            primary="#ffffff",
            primary_variant="#e0e0e0",
            secondary="#ffff00",
            secondary_variant="#ffeb3b",
            text_primary="#ffffff",
            text_secondary="#ffffff",
            text_disabled="#808080",
            border="#ffffff",
            border_hover="#ffff00",
            error="#ff0000",
            success="#00ff00",
            warning="#ffff00",
            info="#00ffff",
            scrollbar_bg="#000000",
            scrollbar_handle="#ffffff",
            scrollbar_handle_hover="#ffff00",
            selection_bg="#ffffff",
            selection_text="#000000",
            tooltip_bg="#ffffff",
            tooltip_text="#000000",
            chart_bg="#000000",
            chart_grid="#808080",
            chart_text="#ffffff",
            is_dark=True
        ),
        
        ColorTheme.HIGH_CONTRAST_LIGHT: ThemeColors(
            # Maximum contrast black on white
            background="#ffffff",
            surface="#ffffff",
            surface_variant="#e0e0e0",
            primary="#000000",
            primary_variant="#1a1a1a",
            secondary="#0000ff",
            secondary_variant="#0000cc",
            text_primary="#000000",
            text_secondary="#000000",
            text_disabled="#808080",
            border="#000000",
            border_hover="#0000ff",
            error="#cc0000",
            success="#008800",
            warning="#cc6600",
            info="#0066cc",
            scrollbar_bg="#ffffff",
            scrollbar_handle="#000000",
            scrollbar_handle_hover="#0000ff",
            selection_bg="#000000",
            selection_text="#ffffff",
            tooltip_bg="#000000",
            tooltip_text="#ffffff",
            chart_bg="#ffffff",
            chart_grid="#808080",
            chart_text="#000000",
            is_dark=False
        ),
        
        ColorTheme.YELLOW_BLACK: ThemeColors(
            # High visibility yellow on black for low vision
            background="#000000",
            surface="#1a1a00",
            surface_variant="#333300",
            primary="#ffff00",
            primary_variant="#ffeb3b",
            secondary="#ffc107",
            secondary_variant="#ffb300",
            text_primary="#ffff00",
            text_secondary="#ffeb3b",
            text_disabled="#808000",
            border="#ffff00",
            border_hover="#ffc107",
            error="#ff5252",
            success="#76ff03",
            warning="#ff9800",
            info="#40c4ff",
            scrollbar_bg="#1a1a00",
            scrollbar_handle="#ffff00",
            scrollbar_handle_hover="#ffc107",
            selection_bg="#ffff00",
            selection_text="#000000",
            tooltip_bg="#ffff00",
            tooltip_text="#000000",
            chart_bg="#0a0a00",
            chart_grid="#808000",
            chart_text="#ffeb3b",
            is_dark=True
        ),
        
        ColorTheme.BLACK_WHITE: ThemeColors(
            # Pure black on white for maximum readability
            background="#ffffff",
            surface="#ffffff",
            surface_variant="#f5f5f5",
            primary="#000000",
            primary_variant="#333333",
            secondary="#666666",
            secondary_variant="#808080",
            text_primary="#000000",
            text_secondary="#333333",
            text_disabled="#999999",
            border="#cccccc",
            border_hover="#999999",
            error="#cc0000",
            success="#008800",
            warning="#cc6600",
            info="#0066cc",
            scrollbar_bg="#f5f5f5",
            scrollbar_handle="#999999",
            scrollbar_handle_hover="#666666",
            selection_bg="#333333",
            selection_text="#ffffff",
            tooltip_bg="#333333",
            tooltip_text="#ffffff",
            chart_bg="#fafafa",
            chart_grid="#cccccc",
            chart_text="#333333",
            is_dark=False
        ),
        
        # Vision-specific themes
        ColorTheme.DEUTERANOPIA_DARK: ThemeColors(
            # Optimized for red-green colorblindness (most common)
            background="#0a0a1a",
            surface="#151525",
            surface_variant="#202035",
            primary="#0099cc",
            primary_variant="#0077aa",
            secondary="#ffcc00",
            secondary_variant="#e6b800",
            text_primary="#e0e8ff",
            text_secondary="#a0b0d0",
            text_disabled="#506080",
            border="#303050",
            border_hover="#404060",
            error="#cc79a7",
            success="#009e73",
            warning="#e69f00",
            info="#56b4e9",
            scrollbar_bg="#151525",
            scrollbar_handle="#404060",
            scrollbar_handle_hover="#505070",
            selection_bg="#0077aa",
            selection_text="#ffffff",
            tooltip_bg="#303050",
            tooltip_text="#e0e8ff",
            chart_bg="#0f0f1f",
            chart_grid="#303050",
            chart_text="#a0b0d0",
            is_dark=True
        ),
        
        ColorTheme.PROTANOPIA_DARK: ThemeColors(
            # Optimized for red weakness
            background="#0a0e1a",
            surface="#151923",
            surface_variant="#202430",
            primary="#0072b2",
            primary_variant="#005a8f",
            secondary="#f0e442",
            secondary_variant="#d4c935",
            text_primary="#e0f0ff",
            text_secondary="#a0c0e0",
            text_disabled="#507090",
            border="#304050",
            border_hover="#405060",
            error="#d55e00",
            success="#009e73",
            warning="#e69f00",
            info="#56b4e9",
            scrollbar_bg="#151923",
            scrollbar_handle="#405060",
            scrollbar_handle_hover="#506070",
            selection_bg="#005a8f",
            selection_text="#ffffff",
            tooltip_bg="#304050",
            tooltip_text="#e0f0ff",
            chart_bg="#0f131f",
            chart_grid="#304050",
            chart_text="#a0c0e0",
            is_dark=True
        ),
        
        ColorTheme.TRITANOPIA_DARK: ThemeColors(
            # Optimized for blue-yellow colorblindness (rare)
            background="#1a0a0a",
            surface="#251515",
            surface_variant="#302020",
            primary="#d55e00",
            primary_variant="#b84c00",
            secondary="#009e73",
            secondary_variant="#007a5a",
            text_primary="#ffe0e0",
            text_secondary="#d0a0a0",
            text_disabled="#806060",
            border="#503030",
            border_hover="#604040",
            error="#cc79a7",
            success="#009e73",
            warning="#e69f00",
            info="#0072b2",
            scrollbar_bg="#251515",
            scrollbar_handle="#604040",
            scrollbar_handle_hover="#705050",
            selection_bg="#b84c00",
            selection_text="#ffffff",
            tooltip_bg="#503030",
            tooltip_text="#ffe0e0",
            chart_bg="#1f0f0f",
            chart_grid="#503030",
            chart_text="#d0a0a0",
            is_dark=True
        ),
        
        ColorTheme.MONOCHROME_DARK: ThemeColors(
            # Pure grayscale dark theme
            background="#0a0a0a",
            surface="#1a1a1a",
            surface_variant="#2a2a2a",
            primary="#d0d0d0",
            primary_variant="#b0b0b0",
            secondary="#808080",
            secondary_variant="#606060",
            text_primary="#f0f0f0",
            text_secondary="#c0c0c0",
            text_disabled="#606060",
            border="#404040",
            border_hover="#505050",
            error="#ffffff",
            success="#e0e0e0",
            warning="#c0c0c0",
            info="#a0a0a0",
            scrollbar_bg="#1a1a1a",
            scrollbar_handle="#505050",
            scrollbar_handle_hover="#606060",
            selection_bg="#b0b0b0",
            selection_text="#000000",
            tooltip_bg="#404040",
            tooltip_text="#f0f0f0",
            chart_bg="#0f0f0f",
            chart_grid="#404040",
            chart_text="#c0c0c0",
            is_dark=True
        ),
        
        ColorTheme.MONOCHROME_LIGHT: ThemeColors(
            # Pure grayscale light theme
            background="#fafafa",
            surface="#ffffff",
            surface_variant="#f0f0f0",
            primary="#303030",
            primary_variant="#404040",
            secondary="#808080",
            secondary_variant="#909090",
            text_primary="#000000",
            text_secondary="#303030",
            text_disabled="#909090",
            border="#c0c0c0",
            border_hover="#b0b0b0",
            error="#000000",
            success="#202020",
            warning="#404040",
            info="#606060",
            scrollbar_bg="#f0f0f0",
            scrollbar_handle="#b0b0b0",
            scrollbar_handle_hover="#a0a0a0",
            selection_bg="#404040",
            selection_text="#ffffff",
            tooltip_bg="#303030",
            tooltip_text="#ffffff",
            chart_bg="#f5f5f5",
            chart_grid="#c0c0c0",
            chart_text="#303030",
            is_dark=False
        ),
    }
    
    def __init__(self):
        super().__init__()
        self._current_theme = ColorTheme.MIDNIGHT_BLUE
    
    def get_theme_names(self) -> List[str]:
        """Get list of available theme names."""
        return [
            ColorTheme.MIDNIGHT_BLUE,
            ColorTheme.FOREST_DARK,
            ColorTheme.VOLCANIC_ASH,
            ColorTheme.DEEP_OCEAN,
            ColorTheme.CYBERPUNK,
            ColorTheme.DARK_AMBER,
            ColorTheme.TWILIGHT_PURPLE,
            ColorTheme.CARBON_GRAY,
            ColorTheme.SOFT_PASTEL,
            ColorTheme.NORDIC_LIGHT,
            ColorTheme.WARM_EARTH,
            ColorTheme.SKY_BLUE,
            ColorTheme.SPRING_MEADOW,
            ColorTheme.CREAM_COFFEE,
            ColorTheme.LAVENDER_MIST,
            ColorTheme.MINT_FRESH,
            ColorTheme.HIGH_CONTRAST_DARK,
            ColorTheme.HIGH_CONTRAST_LIGHT,
            ColorTheme.YELLOW_BLACK,
            ColorTheme.BLACK_WHITE,
            ColorTheme.DEUTERANOPIA_DARK,
            ColorTheme.PROTANOPIA_DARK,
            ColorTheme.TRITANOPIA_DARK,
            ColorTheme.MONOCHROME_DARK,
            ColorTheme.MONOCHROME_LIGHT]
    
    def get_theme_colors(self, theme: str) -> ThemeColors:
        """Get colors for a specific theme."""
        return self.THEMES.get(theme, self.THEMES[ColorTheme.MIDNIGHT_BLUE])
    
    def set_theme(self, theme: str):
        """Set the current theme."""
        if theme in self.THEMES:
            self._current_theme = theme
            self.themeChanged.emit(theme)
    
    def get_current_theme(self) -> str:
        """Get the current theme name."""
        return self._current_theme
    
    def get_current_colors(self) -> ThemeColors:
        """Get colors for the current theme."""
        return self.get_theme_colors(self._current_theme)
    
    def _get_button_text_color(self, colors: ThemeColors) -> str:
        """Get appropriate text color for buttons based on WCAG contrast requirements."""
        # Convert hex to RGB and calculate relative luminance (WCAG formula)
        def get_luminance(hex_color):
            color = hex_color.lstrip('#')
            r, g, b = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            
            # Apply gamma correction
            r = r/12.92 if r <= 0.03928 else ((r + 0.055)/1.055) ** 2.4
            g = g/12.92 if g <= 0.03928 else ((g + 0.055)/1.055) ** 2.4
            b = b/12.92 if b <= 0.03928 else ((b + 0.055)/1.055) ** 2.4
            
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Calculate contrast ratio
        def get_contrast_ratio(color1, color2):
            lum1 = get_luminance(color1)
            lum2 = get_luminance(color2)
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            return (lighter + 0.05) / (darker + 0.05)
        
        # Check contrast with both black and white
        contrast_with_white = get_contrast_ratio(colors.primary, "#ffffff")
        contrast_with_black = get_contrast_ratio(colors.primary, "#000000")
        
        # WCAG AA requires 4.5:1 for normal text, we'll use 3:1 for large bold text
        # If neither meets the standard, choose the better one
        if contrast_with_black >= 3.0:
            return "#000000"
        elif contrast_with_white >= 3.0:
            return "#ffffff"
        else:
            # Neither meets standard, use the one with better contrast
            return "#000000" if contrast_with_black > contrast_with_white else "#ffffff"
    
    def _get_contrast_text_color(self, bg_color: str, prefer_dark: bool = True) -> str:
        """Get appropriate text color for any background based on WCAG contrast requirements."""
        # Convert hex to RGB and calculate relative luminance (WCAG formula)
        def get_luminance(hex_color):
            color = hex_color.lstrip('#')
            r, g, b = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            
            # Apply gamma correction
            r = r/12.92 if r <= 0.03928 else ((r + 0.055)/1.055) ** 2.4
            g = g/12.92 if g <= 0.03928 else ((g + 0.055)/1.055) ** 2.4
            b = b/12.92 if b <= 0.03928 else ((b + 0.055)/1.055) ** 2.4
            
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Calculate contrast ratio
        def get_contrast_ratio(color1, color2):
            lum1 = get_luminance(color1)
            lum2 = get_luminance(color2)
            lighter = max(lum1, lum2)
            darker = min(lum1, lum2)
            return (lighter + 0.05) / (darker + 0.05)
        
        # Check contrast with both black and white
        contrast_with_white = get_contrast_ratio(bg_color, "#ffffff")
        contrast_with_black = get_contrast_ratio(bg_color, "#000000")
        
        # WCAG AA requires 4.5:1 for normal text
        if contrast_with_black >= 4.5:
            return "#000000"
        elif contrast_with_white >= 4.5:
            return "#ffffff"
        else:
            # Neither meets standard, use the one with better contrast
            return "#000000" if contrast_with_black > contrast_with_white else "#ffffff"
    
    def generate_stylesheet(self, theme: str) -> str:
        """Generate a complete Qt stylesheet for the theme."""
        colors = self.get_theme_colors(theme)
        
        return f"""
/* Main Application */
QMainWindow {{
    background-color: {colors.background};
}}

/* Central Widget and containers */
QWidget {{
    background-color: {colors.background};
    color: {colors.text_primary};
}}

/* Tab Widget */
QTabWidget::pane {{
    background-color: {colors.surface};
    border: 1px solid {colors.border};
}}

QTabBar {{
    font-size: 14px;
    font-weight: 500;
}}

QTabBar::tab {{
    background-color: {colors.surface_variant};
    color: {colors.text_secondary};
    padding: 10px 20px;
    margin-right: 2px;
    border: 1px solid {colors.border};
    border-bottom: none;
    outline: none;
}}

QTabBar::tab:selected {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border-color: {colors.primary};
    border-bottom: 2px solid {colors.primary};
    font-weight: 600;
}}

QTabBar::tab:hover {{
    background-color: {colors.surface};
    color: {colors.text_primary};
}}

QTabBar::tab:focus {{
    outline: none;
}}

/* Buttons */
QPushButton {{
    background-color: {colors.primary};
    color: {self._get_button_text_color(colors)};
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {colors.primary_variant};
}}

QPushButton:pressed {{
    background-color: {colors.secondary_variant};
}}

QPushButton:disabled {{
    background-color: {colors.surface_variant};
    color: {colors.text_disabled};
}}

/* Secondary style buttons */
QPushButton[flat="true"] {{
    background-color: transparent;
    color: {colors.primary};
    border: 1px solid {colors.primary};
}}

QPushButton[flat="true"]:hover {{
    background-color: {colors.primary};
    color: {self._get_button_text_color(colors)};
}}

/* ComboBox - DO NOT style arrows to preserve native rendering */
QComboBox {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    padding: 4px 8px;
    border-radius: 4px;
}}

QComboBox:hover {{
    border-color: {colors.primary};
}}

QComboBox QAbstractItemView {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    selection-background-color: {colors.selection_bg};
    selection-color: {self._get_contrast_text_color(colors.selection_bg)};
}}

QComboBox QAbstractItemView::item {{
    padding: 4px;
    background-color: {colors.surface};
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {colors.surface_variant};
    color: {colors.text_primary};
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {colors.selection_bg};
    color: {self._get_contrast_text_color(colors.selection_bg)};
}}

/* ComboBox QListView specific styling */
QComboBox QListView {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    outline: none;
}}

QComboBox QListView::item {{
    padding: 4px;
    background-color: {colors.surface};
}}

QComboBox QListView::item:hover {{
    background-color: {colors.surface_variant};
    color: {colors.text_primary};
}}

QComboBox QListView::item:selected {{
    background-color: {colors.selection_bg};
    color: {self._get_contrast_text_color(colors.selection_bg)};
}}

/* Line Edit */
QLineEdit {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    padding: 6px 12px;
    border-radius: 4px;
}}

QLineEdit:hover {{
    border-color: {colors.border_hover};
}}

QLineEdit:focus {{
    border-color: {colors.primary};
}}

/* Spin Box - Remove border styling to keep native arrows */
QSpinBox, QDoubleSpinBox {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    padding: 4px;
    min-height: 20px;
}}

/* Text Edit */
QTextEdit, QPlainTextEdit {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    border-radius: 4px;
}}

/* List View */
QListView {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    border-radius: 4px;
    outline: none;
}}

QListView::item {{
    padding: 4px;
    border-radius: 2px;
}}

QListView::item:selected {{
    background-color: {colors.selection_bg};
    color: {self._get_contrast_text_color(colors.selection_bg)};
}}

QListView::item:hover {{
    background-color: {colors.surface_variant};
}}

/* Tree View */
QTreeView {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    border-radius: 4px;
    outline: none;
}}

QTreeView::item {{
    padding: 4px;
}}

QTreeView::item:selected {{
    background-color: {colors.selection_bg};
    color: {self._get_contrast_text_color(colors.selection_bg)};
}}

QTreeView::item:hover {{
    background-color: {colors.surface_variant};
}}

/* Table View */
QTableView {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    gridline-color: {colors.border};
}}

QTableView::item:selected {{
    background-color: {colors.selection_bg};
    color: {self._get_contrast_text_color(colors.selection_bg)};
}}

QHeaderView::section {{
    background-color: {colors.surface_variant};
    color: {colors.text_primary};
    padding: 6px;
    border: none;
    border-right: 1px solid {colors.border};
    border-bottom: 1px solid {colors.border};
}}

/* Scroll Bars */
QScrollBar:vertical {{
    background-color: {colors.scrollbar_bg};
    width: 12px;
    border: none;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {colors.scrollbar_handle};
    min-height: 20px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {colors.scrollbar_handle_hover};
}}

QScrollBar:horizontal {{
    background-color: {colors.scrollbar_bg};
    height: 12px;
    border: none;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {colors.scrollbar_handle};
    min-width: 20px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {colors.scrollbar_handle_hover};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    border: none;
    background: none;
}}

/* Menu Bar */
QMenuBar {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border-bottom: 1px solid {colors.border};
}}

QMenuBar::item:selected {{
    background-color: {colors.surface_variant};
}}

/* Menus */
QMenu {{
    background-color: {colors.surface};
    color: {colors.text_primary};
    border: 1px solid {colors.border};
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 20px;
    border-radius: 2px;
}}

QMenu::item:selected {{
    background-color: {colors.selection_bg};
    color: {self._get_contrast_text_color(colors.selection_bg)};
}}

/* Tool Bar */
QToolBar {{
    background-color: {colors.surface};
    border: none;
    padding: 4px;
}}

/* Status Bar */
QStatusBar {{
    background-color: {colors.surface};
    color: {colors.text_secondary};
    border-top: 1px solid {colors.border};
}}

/* Progress Bar */
QProgressBar {{
    background-color: {colors.surface_variant};
    border: 1px solid {colors.border};
    border-radius: 4px;
    text-align: center;
    color: {colors.text_primary};
}}

QProgressBar::chunk {{
    background-color: {colors.primary};
    border-radius: 3px;
}}

/* Slider */
QSlider::groove:horizontal {{
    background-color: {colors.surface_variant};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {colors.primary};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {colors.primary_variant};
}}

/* Check Box */
QCheckBox {{
    color: {colors.text_primary};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {colors.border};
    border-radius: 3px;
    background-color: {colors.surface};
}}

QCheckBox::indicator:checked {{
    background-color: {colors.primary};
    border-color: {colors.primary};
}}

QCheckBox::indicator:hover {{
    border-color: {colors.border_hover};
}}

/* Radio Button */
QRadioButton {{
    color: {colors.text_primary};
    spacing: 8px;
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {colors.border};
    border-radius: 9px;
    background-color: {colors.surface};
}}

QRadioButton::indicator:checked {{
    background-color: {colors.primary};
    border-color: {colors.primary};
}}

/* Group Box */
QGroupBox {{
    background-color: {colors.surface};
    border: 1px solid {colors.border};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
    color: {colors.text_primary};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    background-color: {colors.surface};
}}

/* Tool Tips */
QToolTip {{
    background-color: {colors.tooltip_bg};
    color: {colors.tooltip_text};
    border: 1px solid {colors.border};
    padding: 4px;
}}

/* Dock Widget */
QDockWidget {{
    color: {colors.text_primary};
}}

QDockWidget::title {{
    background-color: {colors.surface_variant};
    padding: 6px;
    border-bottom: 1px solid {colors.border};
}}

/* Labels */
QLabel {{
    color: {colors.text_primary};
}}

QLabel[secondary="true"] {{
    color: {colors.text_secondary};
}}

QLabel[error="true"] {{
    color: {colors.error};
}}

QLabel[success="true"] {{
    color: {colors.success};
}}

QLabel[warning="true"] {{
    color: {colors.warning};
}}

QLabel[info="true"] {{
    color: {colors.info};
}}

/* Splitter */
QSplitter::handle {{
    background-color: {colors.border};
}}

QSplitter::handle:hover {{
    background-color: {colors.border_hover};
}}

/* Canvas/Drawing areas (for matplotlib) */
QWidget[canvas="true"] {{
    background-color: {colors.chart_bg};
    border: 1px solid {colors.border};
}}

/* Message Box - High contrast for visibility */
QMessageBox {{
    background-color: {'#f0f0f0' if colors.is_dark else '#2b2b2b'};
    color: {'#000000' if colors.is_dark else '#ffffff'};
    border: 2px solid {colors.primary};
}}

QMessageBox QLabel {{
    background-color: transparent;
    color: {'#000000' if colors.is_dark else '#ffffff'};
    padding: 10px;
    font-size: 14px;
}}

QMessageBox QPushButton {{
    background-color: {colors.primary};
    color: {self._get_button_text_color(colors)};
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
    min-width: 80px;
    margin: 5px;
}}

QMessageBox QPushButton:hover {{
    background-color: {colors.primary_variant};
}}

QMessageBox QPushButton:pressed {{
    background-color: {colors.secondary_variant};
}}

QMessageBox QPushButton:default {{
    border: 2px solid {colors.primary_variant};
}}

/* Dialog background - High contrast */
QDialog {{
    background-color: {'#e8e8e8' if colors.is_dark else '#2b2b2b'};
    color: {'#000000' if colors.is_dark else '#ffffff'};
    border: 1px solid {colors.border};
}}
"""
