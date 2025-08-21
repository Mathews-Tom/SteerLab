"""
LEARN Interface Implementation

The LEARN mode continuously adapts to user preferences through implicit feedback,
learning from user interactions, corrections, and behavioral patterns to automatically
adjust steering without explicit preference specification.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class FeedbackSignal(Enum):
    """Types of implicit feedback signals."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    EDIT_REQUEST = "edit_request"
    REGENERATION_REQUEST = "regeneration"
    COPY_ACTION = "copy"
    SHARE_ACTION = "share"
    TIME_SPENT = "time_spent"
    FOLLOW_UP_QUESTION = "follow_up"


class LearningMethod(Enum):
    """Different learning approaches."""
    GRADIENT_BASED = "gradient_based"
    PATTERN_MATCHING = "pattern_matching"
    PREFERENCE_INFERENCE = "preference_inference"
    COLLABORATIVE_FILTERING = "collaborative_filtering"


@dataclass
class InteractionEvent:
    """Represents a user interaction with the system."""
    event_id: str
    timestamp: datetime
    prompt: str
    response: str
    preferences_used: Dict[str, float]
    feedback_signal: Optional[FeedbackSignal] = None
    feedback_strength: float = 1.0  # 0.0 to 1.0
    user_edit: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningUpdate:
    """Represents a preference update based on learning."""
    preference_name: str
    old_value: float
    new_value: float
    confidence: float
    reasoning: str
    evidence_events: List[str]  # Event IDs that led to this update


class LearnInterface:
    """
    Implements the LEARN interaction mode for adaptive preference learning.

    LEARN mode automatically adapts to user preferences through implicit feedback,
    analyzing user behavior patterns to continuously refine preference settings
    without requiring explicit user input.
    """

    def __init__(
        self,
        preference_names: List[str],
        learning_rate: float = 0.05,
        memory_window: int = 100,
        confidence_threshold: float = 0.7,
        update_frequency: int = 5
    ):
        """
        Initialize the LEARN interface.

        Args:
            preference_names: List of preferences to learn
            learning_rate: Rate of preference adaptation
            memory_window: Number of recent interactions to consider
            confidence_threshold: Minimum confidence for preference updates
            update_frequency: Update preferences every N interactions
        """
        self.preference_names = preference_names
        self.learning_rate = learning_rate
        self.memory_window = memory_window
        self.confidence_threshold = confidence_threshold
        self.update_frequency = update_frequency

        # Current learned preferences
        self.learned_preferences = {name: 0.0 for name in preference_names}
        self.preference_confidence = {name: 0.0 for name in preference_names}

        # Interaction history
        self.interaction_history = deque(maxlen=memory_window)
        self.learning_updates = []

        # Learning statistics
        self.total_interactions = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
        self.last_update_interaction = 0

        # Pattern detection
        self.feedback_patterns = defaultdict(list)
        self.context_patterns = defaultdict(list)

    def record_interaction(
        self,
        prompt: str,
        response: str,
        preferences_used: Dict[str, float],
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a new interaction for learning.

        Args:
            prompt: User's input prompt
            response: Generated response
            preferences_used: Preferences that were applied
            session_id: Optional session identifier
            context: Additional context information

        Returns:
            Event ID for this interaction
        """
        event_id = f"event_{self.total_interactions + 1}_{datetime.now().timestamp()}"

        event = InteractionEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            prompt=prompt,
            response=response,
            preferences_used=preferences_used.copy(),
            session_id=session_id,
            context=context or {}
        )

        self.interaction_history.append(event)
        self.total_interactions += 1

        # Trigger learning if enough interactions have passed
        if (self.total_interactions - self.last_update_interaction) >= self.update_frequency:
            self._update_preferences()

        logger.info(f"Recorded interaction {event_id}")
        return event_id

    def record_feedback(
        self,
        event_id: str,
        feedback_signal: FeedbackSignal,
        feedback_strength: float = 1.0,
        user_edit: Optional[str] = None
    ) -> bool:
        """
        Record user feedback on a specific interaction.

        Args:
            event_id: ID of the interaction event
            feedback_signal: Type of feedback
            feedback_strength: Strength of the feedback (0.0 to 1.0)
            user_edit: Optional user edit/correction

        Returns:
            True if feedback was recorded successfully
        """
        # Find the event
        event = None
        for interaction in self.interaction_history:
            if interaction.event_id == event_id:
                event = interaction
                break

        if not event:
            logger.warning(f"Event {event_id} not found")
            return False

        # Update event with feedback
        event.feedback_signal = feedback_signal
        event.feedback_strength = feedback_strength
        event.user_edit = user_edit

        # Update statistics
        if feedback_signal in [FeedbackSignal.THUMBS_UP, FeedbackSignal.COPY_ACTION, FeedbackSignal.SHARE_ACTION]:
            self.positive_feedback_count += 1
        elif feedback_signal in [FeedbackSignal.THUMBS_DOWN, FeedbackSignal.EDIT_REQUEST, FeedbackSignal.REGENERATION_REQUEST]:
            self.negative_feedback_count += 1

        # Store pattern
        self.feedback_patterns[feedback_signal].append({
            "event_id": event_id,
            "preferences": event.preferences_used.copy(),
            "context": event.context.copy(),
            "timestamp": event.timestamp
        })

        logger.info(f"Recorded feedback {feedback_signal} for event {event_id}")

        # Immediate learning update for strong negative feedback
        if feedback_signal in [FeedbackSignal.THUMBS_DOWN, FeedbackSignal.EDIT_REQUEST] and feedback_strength > 0.7:
            self._learn_from_negative_feedback(event)

        return True

    def get_current_preferences(self) -> Dict[str, float]:
        """Get current learned preferences."""
        return self.learned_preferences.copy()

    def get_preference_confidence(self) -> Dict[str, float]:
        """Get confidence scores for learned preferences."""
        return self.preference_confidence.copy()

    def _update_preferences(self) -> List[LearningUpdate]:
        """Update preferences based on accumulated learning."""
        updates = []

        # Analyze recent interactions
        recent_interactions = list(self.interaction_history)[-self.update_frequency:]

        # Learn from explicit feedback
        updates.extend(self._learn_from_explicit_feedback(recent_interactions))

        # Learn from patterns
        updates.extend(self._learn_from_patterns())

        # Learn from user edits
        updates.extend(self._learn_from_user_edits(recent_interactions))

        # Apply updates
        for update in updates:
            if update.confidence >= self.confidence_threshold:
                self._apply_learning_update(update)

        self.last_update_interaction = self.total_interactions

        if updates:
            logger.info(f"Applied {len(updates)} preference updates")

        return updates

    def _learn_from_explicit_feedback(self, interactions: List[InteractionEvent]) -> List[LearningUpdate]:
        """Learn from explicit thumbs up/down feedback."""
        updates = []

        for interaction in interactions:
            if interaction.feedback_signal in [FeedbackSignal.THUMBS_UP, FeedbackSignal.THUMBS_DOWN]:
                for pref_name in self.preference_names:
                    if pref_name in interaction.preferences_used:
                        current_value = interaction.preferences_used[pref_name]

                        if interaction.feedback_signal == FeedbackSignal.THUMBS_UP:
                            # Positive feedback: move toward the used preference value
                            target_value = current_value * 1.1  # Amplify slightly
                            reasoning = f"Positive feedback on response with {pref_name}={current_value}"
                        else:
                            # Negative feedback: move away from the used preference value
                            target_value = current_value * 0.7  # Reduce
                            reasoning = f"Negative feedback on response with {pref_name}={current_value}"

                        target_value = max(-1.0, min(1.0, target_value))

                        updates.append(LearningUpdate(
                            preference_name=pref_name,
                            old_value=self.learned_preferences[pref_name],
                            new_value=target_value,
                            confidence=interaction.feedback_strength,
                            reasoning=reasoning,
                            evidence_events=[interaction.event_id]
                        ))

        return updates

    def _learn_from_patterns(self) -> List[LearningUpdate]:
        """Learn from behavioral patterns."""
        updates = []

        # Analyze copy/share patterns (indicate preferred content)
        positive_actions = [FeedbackSignal.COPY_ACTION, FeedbackSignal.SHARE_ACTION]
        positive_preferences = self._extract_preference_patterns(positive_actions)

        for pref_name, (avg_value, confidence, event_ids) in positive_preferences.items():
            if confidence > self.confidence_threshold:
                updates.append(LearningUpdate(
                    preference_name=pref_name,
                    old_value=self.learned_preferences[pref_name],
                    new_value=avg_value,
                    confidence=confidence,
                    reasoning=f"User frequently copies/shares content with {pref_name}≈{avg_value:.2f}",
                    evidence_events=event_ids
                ))

        return updates

    def _learn_from_user_edits(self, interactions: List[InteractionEvent]) -> List[LearningUpdate]:
        """Learn from user edits and corrections."""
        updates = []

        for interaction in interactions:
            if interaction.user_edit and interaction.feedback_signal == FeedbackSignal.EDIT_REQUEST:
                # Analyze the edit to infer preference changes
                edit_analysis = self._analyze_user_edit(interaction.response, interaction.user_edit)

                for pref_name, preference_change in edit_analysis.items():
                    if pref_name in self.preference_names:
                        current_pref = interaction.preferences_used.get(pref_name, 0.0)
                        suggested_value = current_pref + preference_change
                        suggested_value = max(-1.0, min(1.0, suggested_value))

                        updates.append(LearningUpdate(
                            preference_name=pref_name,
                            old_value=self.learned_preferences[pref_name],
                            new_value=suggested_value,
                            confidence=0.6,  # Medium confidence from edit analysis
                            reasoning=f"User edit suggests {pref_name} change: {preference_change:+.2f}",
                            evidence_events=[interaction.event_id]
                        ))

        return updates

    def _learn_from_negative_feedback(self, event: InteractionEvent):
        """Immediate learning from strong negative feedback."""
        # For negative feedback, reduce the strength of preferences that were used
        for pref_name in self.preference_names:
            if pref_name in event.preferences_used:
                current_value = event.preferences_used[pref_name]
                if abs(current_value) > 0.1:  # Only adjust non-neutral preferences
                    # Reduce the preference strength
                    new_value = current_value * 0.8
                    self.learned_preferences[pref_name] = (
                        self.learned_preferences[pref_name] +
                        self.learning_rate * (new_value - self.learned_preferences[pref_name])
                    )

                    logger.info(f"Immediate adjustment: {pref_name} {current_value:.2f} -> {new_value:.2f}")

    def _extract_preference_patterns(self, feedback_signals: List[FeedbackSignal]) -> Dict[str, Tuple[float, float, List[str]]]:
        """Extract preference patterns from specific feedback signals."""
        patterns = {}

        for signal in feedback_signals:
            if signal in self.feedback_patterns:
                signal_data = self.feedback_patterns[signal]

                # Aggregate preferences for this signal type
                pref_values = defaultdict(list)
                event_ids = []

                for data in signal_data[-20:]:  # Recent 20 events
                    event_ids.append(data["event_id"])
                    for pref_name, value in data["preferences"].items():
                        pref_values[pref_name].append(value)

                # Calculate averages and confidence
                for pref_name, values in pref_values.items():
                    if len(values) >= 3:  # Minimum evidence
                        avg_value = sum(values) / len(values)
                        confidence = min(1.0, len(values) / 10.0)  # More evidence = higher confidence
                        patterns[pref_name] = (avg_value, confidence, event_ids)

        return patterns

    def _analyze_user_edit(self, original_response: str, user_edit: str) -> Dict[str, float]:
        """
        Analyze user edits to infer preference changes.

        This is a simplified version - in practice, you might use NLP
        to analyze the semantic differences.
        """
        preference_changes = {}

        # Simple heuristics for preference inference

        # Formality analysis
        if len(user_edit) > len(original_response) * 1.2:
            preference_changes["detail_level"] = 0.2
        elif len(user_edit) < len(original_response) * 0.8:
            preference_changes["detail_level"] = -0.2

        # Check for formal language markers
        formal_markers = ["please", "would", "could", "kindly", "respectfully"]
        informal_markers = ["hey", "gonna", "wanna", "yeah", "cool"]

        original_formal_count = sum(1 for marker in formal_markers if marker.lower() in original_response.lower())
        edit_formal_count = sum(1 for marker in formal_markers if marker.lower() in user_edit.lower())

        if edit_formal_count > original_formal_count:
            preference_changes["formality"] = 0.3
        elif edit_formal_count < original_formal_count:
            preference_changes["formality"] = -0.3

        return preference_changes

    def _apply_learning_update(self, update: LearningUpdate):
        """Apply a learning update to preferences."""
        # Weighted update based on confidence
        old_value = self.learned_preferences[update.preference_name]
        new_value = update.new_value

        # Apply learning rate and confidence weighting
        adjusted_value = (
            old_value +
            self.learning_rate * update.confidence * (new_value - old_value)
        )

        self.learned_preferences[update.preference_name] = max(-1.0, min(1.0, adjusted_value))

        # Update confidence
        self.preference_confidence[update.preference_name] = min(
            1.0,
            self.preference_confidence[update.preference_name] + update.confidence * 0.1
        )

        # Store the update
        self.learning_updates.append(update)

        logger.info(
            f"Updated {update.preference_name}: {old_value:.3f} -> "
            f"{self.learned_preferences[update.preference_name]:.3f} "
            f"(confidence: {update.confidence:.2f})"
        )

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        total_feedback = self.positive_feedback_count + self.negative_feedback_count

        return {
            "total_interactions": self.total_interactions,
            "total_feedback": total_feedback,
            "positive_feedback": self.positive_feedback_count,
            "negative_feedback": self.negative_feedback_count,
            "feedback_ratio": (
                self.positive_feedback_count / max(1, total_feedback)
            ),
            "learning_updates": len(self.learning_updates),
            "avg_confidence": (
                sum(self.preference_confidence.values()) /
                len(self.preference_confidence)
            ),
            "preferences_learned": {
                name: conf > 0.5
                for name, conf in self.preference_confidence.items()
            }
        }

    def export_learned_preferences(self) -> Dict[str, Any]:
        """Export learned preferences and metadata."""
        return {
            "learned_preferences": self.learned_preferences.copy(),
            "preference_confidence": self.preference_confidence.copy(),
            "learning_statistics": self.get_learning_statistics(),
            "recent_updates": [
                {
                    "preference": update.preference_name,
                    "old_value": update.old_value,
                    "new_value": update.new_value,
                    "confidence": update.confidence,
                    "reasoning": update.reasoning
                }
                for update in self.learning_updates[-10:]  # Last 10 updates
            ]
        }

    def reset_learning(self) -> None:
        """Reset all learned preferences and history."""
        self.learned_preferences = {name: 0.0 for name in self.preference_names}
        self.preference_confidence = {name: 0.0 for name in self.preference_names}
        self.interaction_history.clear()
        self.learning_updates.clear()
        self.feedback_patterns.clear()
        self.context_patterns.clear()

        self.total_interactions = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
        self.last_update_interaction = 0

        logger.info("Learning state reset")