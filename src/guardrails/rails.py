# src/guardrails/rails.py
from nemoguardrails import RailsConfig, LLMRails
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GuardrailsManager:
    def __init__(self):
        config_path = Path(__file__).parent / "colang"
        self.config = RailsConfig.from_path(str(config_path))
        self.rails = LLMRails(self.config)
    
    async def check_input(self, user_message: str) -> tuple[bool, str | None]:
        """
        Check if user input passes guardrails.
        Returns (passed, blocked_reason).
        """
        try:
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": user_message}]
            )
            
            # Check if the response indicates blocking
            response_text = response.get("content", "")
            
            blocked_indicators = [
                "I can't help with that",
                "I'm specialized in YouTube",
                "refuse",
            ]
            
            for indicator in blocked_indicators:
                if indicator.lower() in response_text.lower():
                    return False, response_text
            
            return True, None
            
        except Exception as e:
            logger.error(f"Guardrails check failed: {e}")
            # Fail open for availability, but log for review
            return True, None
    
    async def check_output(self, response: str) -> tuple[bool, str]:
        """Validate LLM output before sending to user."""
        try:
            result = await self.rails.generate_async(
                messages=[
                    {"role": "assistant", "content": response}
                ]
            )
            return True, result.get("content", response)
        except Exception as e:
            logger.error(f"Output guardrails failed: {e}")
            return True, response


guardrails_manager = GuardrailsManager()
