"""
CALIBRATE Interface Implementation

The CALIBRATE mode helps users discover their preferences through an interactive
process of generating examples and getting user feedback to iteratively converge
on optimal preference settings.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CalibrationStep(Enum):
    """Represents different steps in the calibration process."""

    INITIALIZE = "initialize"
    PRESENT_OPTIONS = "present_options"
    COLLECT_FEEDBACK = "collect_feedback"
    UPDATE_PREFERENCES = "update_preferences"
    CONVERGED = "converged"


class FeedbackType(Enum):
    """Types of feedback users can provide."""

    PREFER_A = "prefer_a"
    PREFER_B = "prefer_b"
    NEUTRAL = "neutral"
    TOO_EXTREME = "too_extreme"
    NOT_ENOUGH = "not_enough"


@dataclass
class CalibrationSample:
    """Represents a sample generated during calibration."""

    prompt: str
    response: str
    preferences: Dict[str, float]
    sample_id: str


@dataclass
class CalibrationPair:
    """A pair of samples for comparison."""

    sample_a: CalibrationSample
    sample_b: CalibrationSample
    pair_id: str


@dataclass
class UserFeedback:
    """User feedback on a calibration pair or sample."""

    pair_id: Optional[str]
    sample_id: Optional[str]
    feedback_type: FeedbackType
    confidence: float = 1.0  # 0.0 to 1.0
    notes: Optional[str] = None


class CalibrateInterface:
    """
    Implements the CALIBRATE interaction mode for preference discovery.

    CALIBRATE mode uses an iterative process to help users discover their
    preferences by presenting them with examples and collecting feedback
    to refine preference settings.
    """

    def __init__(
        self,
        preference_names: List[str],
        initial_range: float = 0.5,
        convergence_threshold: float = 0.1,
        max_iterations: int = 10,
    ):
        """
        Initialize the CALIBRATE interface.

        Args:
            preference_names: List of preferences to calibrate
            initial_range: Initial range for preference exploration
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum number of calibration iterations
        """
        self.preference_names = preference_names
        self.initial_range = initial_range
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        # Current state
        self.current_step = CalibrationStep.INITIALIZE
        self.iteration = 0
        self.current_preferences = {name: 0.0 for name in preference_names}
        self.preference_ranges = {name: initial_range for name in preference_names}

        # History
        self.samples_generated = []
        self.pairs_presented = []
        self.feedback_received = []
        self.preference_history = []

        # Active calibration data
        self.current_pair: Optional[CalibrationPair] = None
        self.calibration_prompts = [
            "Tell me about your ideal vacation.",
            "Explain the benefits of renewable energy.",
            "Write a brief introduction about yourself.",
            "Describe your favorite hobby.",
            "Give advice on time management.",
        ]

    def start_calibration(self, initial_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Start the calibration process.

        Args:
            initial_prompt: Optional custom prompt for calibration

        Returns:
            Dictionary with calibration status and next steps
        """
        self.current_step = CalibrationStep.INITIALIZE
        self.iteration = 0

        logger.info("Starting preference calibration")

        # Reset state
        self.current_preferences = {name: 0.0 for name in self.preference_names}
        self.preference_ranges = {
            name: self.initial_range for name in self.preference_names
        }
        self.samples_generated.clear()
        self.pairs_presented.clear()
        self.feedback_received.clear()
        self.preference_history.clear()

        if initial_prompt:
            self.calibration_prompts.insert(0, initial_prompt)

        return self.get_next_step()

    def get_next_step(self) -> Dict[str, Any]:
        """
        Get the next step in the calibration process.

        Returns:
            Dictionary describing the next step and required actions
        """
        if self.current_step == CalibrationStep.INITIALIZE:
            return self._initialize_calibration()
        elif self.current_step == CalibrationStep.PRESENT_OPTIONS:
            return self._present_options()
        elif self.current_step == CalibrationStep.COLLECT_FEEDBACK:
            return self._collect_feedback()
        elif self.current_step == CalibrationStep.UPDATE_PREFERENCES:
            return self._update_preferences()
        elif self.current_step == CalibrationStep.CONVERGED:
            return self._finalize_calibration()
        else:
            raise ValueError(f"Unknown calibration step: {self.current_step}")

    def _initialize_calibration(self) -> Dict[str, Any]:
        """Initialize the calibration process."""
        self.current_step = CalibrationStep.PRESENT_OPTIONS
        self.iteration = 1

        return {
            "step": "initialize",
            "message": "Starting preference calibration. We'll show you pairs of responses and ask for your feedback.",
            "preferences_to_calibrate": self.preference_names,
            "estimated_steps": self.max_iterations,
            "next_action": "present_options",
        }

    def _present_options(self) -> Dict[str, Any]:
        """Present options for user comparison."""
        # Generate preference variations for comparison
        preference_a, preference_b = self._generate_preference_pair()

        # Select a prompt for this iteration
        prompt = self._select_calibration_prompt()

        # Create sample pair
        sample_a = CalibrationSample(
            prompt=prompt,
            response="[Response A would be generated here with preferences]",
            preferences=preference_a,
            sample_id=f"sample_a_{self.iteration}",
        )

        sample_b = CalibrationSample(
            prompt=prompt,
            response="[Response B would be generated here with preferences]",
            preferences=preference_b,
            sample_id=f"sample_b_{self.iteration}",
        )

        self.current_pair = CalibrationPair(
            sample_a=sample_a, sample_b=sample_b, pair_id=f"pair_{self.iteration}"
        )

        self.pairs_presented.append(self.current_pair)
        self.current_step = CalibrationStep.COLLECT_FEEDBACK

        return {
            "step": "present_options",
            "iteration": self.iteration,
            "prompt": prompt,
            "option_a": {
                "id": sample_a.sample_id,
                "response": sample_a.response,
                "preferences": sample_a.preferences,
            },
            "option_b": {
                "id": sample_b.sample_id,
                "response": sample_b.response,
                "preferences": sample_b.preferences,
            },
            "feedback_options": [
                {"value": "prefer_a", "label": "I prefer option A"},
                {"value": "prefer_b", "label": "I prefer option B"},
                {"value": "neutral", "label": "Both are similar/equally good"},
                {"value": "too_extreme", "label": "Both are too extreme"},
                {"value": "not_enough", "label": "Both are not strong enough"},
            ],
            "next_action": "collect_feedback",
        }

    def _collect_feedback(self) -> Dict[str, Any]:
        """Wait for and validate user feedback."""
        return {
            "step": "collect_feedback",
            "message": "Please provide your feedback on the presented options.",
            "waiting_for": "user_feedback",
            "current_pair_id": self.current_pair.pair_id if self.current_pair else None,
        }

    def process_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """
        Process user feedback and update preferences.

        Args:
            feedback: User feedback on the current pair

        Returns:
            Dictionary with processing results and next steps
        """
        if self.current_step != CalibrationStep.COLLECT_FEEDBACK:
            return {"error": "Not currently collecting feedback"}

        if not self.current_pair:
            return {"error": "No active pair for feedback"}

        # Validate feedback
        if feedback.pair_id != self.current_pair.pair_id:
            return {"error": "Feedback pair_id doesn't match current pair"}

        # Store feedback
        self.feedback_received.append(feedback)
        logger.info(
            f"Received feedback: {feedback.feedback_type} for pair {feedback.pair_id}"
        )

        # Move to update step
        self.current_step = CalibrationStep.UPDATE_PREFERENCES
        return self._update_preferences()

    def _update_preferences(self) -> Dict[str, Any]:
        """Update preferences based on received feedback."""
        if not self.feedback_received or not self.current_pair:
            return {"error": "No feedback to process"}

        latest_feedback = self.feedback_received[-1]

        # Store current preferences before update
        old_preferences = self.current_preferences.copy()

        # Update preferences based on feedback type
        if latest_feedback.feedback_type == FeedbackType.PREFER_A:
            self._move_towards_preferences(self.current_pair.sample_a.preferences)
        elif latest_feedback.feedback_type == FeedbackType.PREFER_B:
            self._move_towards_preferences(self.current_pair.sample_b.preferences)
        elif latest_feedback.feedback_type == FeedbackType.TOO_EXTREME:
            self._reduce_preference_strength()
        elif latest_feedback.feedback_type == FeedbackType.NOT_ENOUGH:
            self._increase_preference_strength()
        # NEUTRAL feedback doesn't change preferences but reduces search range

        # Update search ranges (binary search approach)
        self._update_search_ranges(latest_feedback)

        # Store preference change
        self.preference_history.append(
            {
                "iteration": self.iteration,
                "old_preferences": old_preferences,
                "new_preferences": self.current_preferences.copy(),
                "feedback": latest_feedback.feedback_type.value,
                "ranges": self.preference_ranges.copy(),
            }
        )

        # Check convergence
        if self._check_convergence() or self.iteration >= self.max_iterations:
            self.current_step = CalibrationStep.CONVERGED
            return self._finalize_calibration()
        else:
            # Continue calibration
            self.iteration += 1
            self.current_step = CalibrationStep.PRESENT_OPTIONS
            return self.get_next_step()

    def _generate_preference_pair(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate a pair of preference settings for comparison."""
        preference_a = {}
        preference_b = {}

        for pref_name in self.preference_names:
            current_value = self.current_preferences[pref_name]
            range_size = self.preference_ranges[pref_name]

            # Generate two values around the current preference
            offset = range_size / 2
            preference_a[pref_name] = max(-1.0, min(1.0, current_value - offset))
            preference_b[pref_name] = max(-1.0, min(1.0, current_value + offset))

        return preference_a, preference_b

    def _select_calibration_prompt(self) -> str:
        """Select an appropriate prompt for calibration."""
        if self.iteration <= len(self.calibration_prompts):
            return self.calibration_prompts[self.iteration - 1]
        else:
            return random.choice(self.calibration_prompts)

    def _move_towards_preferences(self, target_preferences: Dict[str, float]):
        """Move current preferences toward target preferences."""
        learning_rate = 0.3  # How much to move toward target

        for pref_name in self.preference_names:
            if pref_name in target_preferences:
                current = self.current_preferences[pref_name]
                target = target_preferences[pref_name]
                self.current_preferences[pref_name] = current + learning_rate * (
                    target - current
                )

    def _reduce_preference_strength(self):
        """Reduce the strength of all preferences."""
        damping_factor = 0.7
        for pref_name in self.preference_names:
            self.current_preferences[pref_name] *= damping_factor

    def _increase_preference_strength(self):
        """Increase the strength of preferences."""
        amplification_factor = 1.3
        for pref_name in self.preference_names:
            current = self.current_preferences[pref_name]
            if current != 0:
                self.current_preferences[pref_name] = max(
                    -1.0, min(1.0, current * amplification_factor)
                )

    def _update_search_ranges(self, feedback: UserFeedback):
        """Update search ranges based on feedback (binary search approach)."""
        range_reduction = 0.7

        for pref_name in self.preference_names:
            self.preference_ranges[pref_name] *= range_reduction
            # Don't let range get too small
            self.preference_ranges[pref_name] = max(
                0.05, self.preference_ranges[pref_name]
            )

    def _check_convergence(self) -> bool:
        """Check if preferences have converged."""
        # Check if all ranges are below threshold
        for range_val in self.preference_ranges.values():
            if range_val > self.convergence_threshold:
                return False
        return True

    def _finalize_calibration(self) -> Dict[str, Any]:
        """Finalize the calibration process."""
        return {
            "step": "converged",
            "message": "Calibration complete! Your preferences have been determined.",
            "final_preferences": self.current_preferences.copy(),
            "iterations_completed": self.iteration,
            "confidence_scores": {
                name: 1.0 - (range_val / self.initial_range)
                for name, range_val in self.preference_ranges.items()
            },
            "calibration_summary": {
                "total_feedback": len(self.feedback_received),
                "preference_changes": len(self.preference_history),
                "converged": self._check_convergence(),
            },
        }

    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        return {
            "current_step": self.current_step.value,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "current_preferences": self.current_preferences.copy(),
            "preference_ranges": self.preference_ranges.copy(),
            "progress": min(1.0, self.iteration / self.max_iterations),
            "is_converged": self._check_convergence(),
        }

    def reset_calibration(self) -> Dict[str, Any]:
        """Reset calibration to start over."""
        return self.start_calibration()
