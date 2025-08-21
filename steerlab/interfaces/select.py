"""
SELECT Interface Implementation

The SELECT mode allows users to directly specify their preferences using sliders or
discrete choices. This is the most straightforward interaction mode where users
have explicit control over preference strengths.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreferenceOption:
    """Represents a single preference option in SELECT mode."""
    name: str
    description: str
    min_value: float = -1.0
    max_value: float = 1.0
    default_value: float = 0.0
    step: float = 0.1
    category: Optional[str] = None


@dataclass
class SelectConfig:
    """Configuration for SELECT mode interface."""
    available_preferences: List[PreferenceOption]
    allow_custom_preferences: bool = False
    max_active_preferences: Optional[int] = None
    ui_style: str = "sliders"  # "sliders", "buttons", "dropdown"


class SelectInterface:
    """
    Implements the SELECT interaction mode for preference-based steering.

    In SELECT mode, users directly choose their preferences through explicit
    controls like sliders, buttons, or dropdowns. This provides immediate
    and predictable control over model behavior.
    """

    def __init__(self, config: SelectConfig):
        """
        Initialize the SELECT interface.

        Args:
            config: Configuration for available preferences and UI options
        """
        self.config = config
        self.current_preferences = {}
        self.preference_history = []

    def get_available_preferences(self) -> List[PreferenceOption]:
        """Get list of available preferences for selection."""
        return self.config.available_preferences

    def set_preference(self, preference_name: str, value: float) -> bool:
        """
        Set a specific preference value.

        Args:
            preference_name: Name of the preference to set
            value: Preference value (-1.0 to 1.0)

        Returns:
            True if preference was set successfully, False otherwise
        """
        # Find the preference option
        preference_option = None
        for option in self.config.available_preferences:
            if option.name == preference_name:
                preference_option = option
                break

        if preference_option is None:
            if not self.config.allow_custom_preferences:
                logger.warning(f"Unknown preference: {preference_name}")
                return False
            # Create a default preference option for custom preferences
            preference_option = PreferenceOption(
                name=preference_name,
                description=f"Custom preference: {preference_name}"
            )

        # Validate value range
        if not (preference_option.min_value <= value <= preference_option.max_value):
            logger.warning(
                f"Preference value {value} out of range "
                f"[{preference_option.min_value}, {preference_option.max_value}]"
            )
            return False

        # Check max active preferences limit
        if (self.config.max_active_preferences and
            len(self.current_preferences) >= self.config.max_active_preferences and
            preference_name not in self.current_preferences):
            logger.warning(
                f"Maximum active preferences ({self.config.max_active_preferences}) reached"
            )
            return False

        # Set the preference
        old_value = self.current_preferences.get(preference_name, 0.0)
        self.current_preferences[preference_name] = value

        # Log the change
        self.preference_history.append({
            "preference": preference_name,
            "old_value": old_value,
            "new_value": value,
            "action": "set"
        })

        logger.info(f"Set preference {preference_name} = {value}")
        return True

    def remove_preference(self, preference_name: str) -> bool:
        """
        Remove a preference from the active set.

        Args:
            preference_name: Name of the preference to remove

        Returns:
            True if preference was removed, False if it wasn't active
        """
        if preference_name not in self.current_preferences:
            return False

        old_value = self.current_preferences.pop(preference_name)

        # Log the change
        self.preference_history.append({
            "preference": preference_name,
            "old_value": old_value,
            "new_value": 0.0,
            "action": "remove"
        })

        logger.info(f"Removed preference {preference_name}")
        return True

    def get_current_preferences(self) -> Dict[str, float]:
        """Get current active preferences."""
        return self.current_preferences.copy()

    def reset_preferences(self) -> None:
        """Reset all preferences to default values."""
        old_preferences = self.current_preferences.copy()
        self.current_preferences.clear()

        # Log the reset
        self.preference_history.append({
            "action": "reset",
            "old_preferences": old_preferences,
            "new_preferences": {}
        })

        logger.info("Reset all preferences")

    def set_multiple_preferences(self, preferences: Dict[str, float]) -> Dict[str, bool]:
        """
        Set multiple preferences at once.

        Args:
            preferences: Dictionary of preference names to values

        Returns:
            Dictionary indicating success/failure for each preference
        """
        results = {}
        for name, value in preferences.items():
            results[name] = self.set_preference(name, value)
        return results

    def get_preference_constraints(self, preference_name: str) -> Optional[Dict[str, Any]]:
        """
        Get constraints for a specific preference.

        Args:
            preference_name: Name of the preference

        Returns:
            Dictionary with constraints (min, max, step) or None if not found
        """
        for option in self.config.available_preferences:
            if option.name == preference_name:
                return {
                    "min_value": option.min_value,
                    "max_value": option.max_value,
                    "default_value": option.default_value,
                    "step": option.step,
                    "description": option.description,
                    "category": option.category
                }
        return None

    def get_preference_suggestions(self, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get preference suggestions based on context.

        Args:
            context: Optional context for suggestions (e.g., "formal_writing")

        Returns:
            List of suggested preference configurations
        """
        suggestions = []

        # Default suggestions
        if context == "formal_writing":
            suggestions.append({
                "name": "Formal Communication",
                "description": "Suitable for business and academic writing",
                "preferences": {"formality": 0.8, "creativity": -0.2}
            })
        elif context == "creative_writing":
            suggestions.append({
                "name": "Creative Expression",
                "description": "Encourages creative and imaginative responses",
                "preferences": {"creativity": 0.8, "formality": -0.3}
            })
        elif context == "friendly_chat":
            suggestions.append({
                "name": "Casual Conversation",
                "description": "Relaxed and friendly communication style",
                "preferences": {"formality": -0.5, "friendliness": 0.7}
            })
        else:
            # General suggestions based on available preferences
            if any(p.name == "formality" for p in self.config.available_preferences):
                suggestions.append({
                    "name": "Balanced",
                    "description": "Neutral preference settings",
                    "preferences": {"formality": 0.0}
                })
                suggestions.append({
                    "name": "Professional",
                    "description": "More formal communication",
                    "preferences": {"formality": 0.6}
                })
                suggestions.append({
                    "name": "Casual",
                    "description": "More informal communication",
                    "preferences": {"formality": -0.4}
                })

        return suggestions

    def export_preferences(self) -> Dict[str, Any]:
        """
        Export current preferences and configuration.

        Returns:
            Dictionary containing all preference data
        """
        return {
            "current_preferences": self.current_preferences,
            "available_preferences": [
                {
                    "name": p.name,
                    "description": p.description,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "default_value": p.default_value,
                    "step": p.step,
                    "category": p.category
                }
                for p in self.config.available_preferences
            ],
            "history": self.preference_history[-10:]  # Last 10 changes
        }

    def import_preferences(self, data: Dict[str, Any]) -> bool:
        """
        Import preferences from exported data.

        Args:
            data: Previously exported preference data

        Returns:
            True if import was successful
        """
        try:
            if "current_preferences" in data:
                results = self.set_multiple_preferences(data["current_preferences"])
                return all(results.values())
            return True
        except Exception as e:
            logger.error(f"Failed to import preferences: {e}")
            return False


def create_default_select_interface() -> SelectInterface:
    """Create a SELECT interface with common preference options."""
    default_preferences = [
        PreferenceOption(
            name="formality",
            description="Communication formality level",
            category="Style"
        ),
        PreferenceOption(
            name="creativity",
            description="Creative vs conservative responses",
            category="Style"
        ),
        PreferenceOption(
            name="detail_level",
            description="Amount of detail in responses",
            category="Content"
        ),
        PreferenceOption(
            name="confidence",
            description="Confidence vs uncertainty in statements",
            category="Tone"
        ),
        PreferenceOption(
            name="friendliness",
            description="Warmth and friendliness of tone",
            category="Tone"
        )
    ]

    config = SelectConfig(
        available_preferences=default_preferences,
        allow_custom_preferences=True,
        max_active_preferences=5,
        ui_style="sliders"
    )

    return SelectInterface(config)