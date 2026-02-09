from typing import Optional, Dict

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.providers import get_provider, BaseProvider

logger = get_logger("AuraRouter.Fabric")


class ComputeFabric:
    """N-model routing with graceful degradation.

    Iterates through a role's model chain until one provider returns a
    valid response, then stops.
    """

    def __init__(self, config: ConfigLoader):
        self._config = config
        self._provider_cache: Dict[str, BaseProvider] = {}

    def execute(
        self, role: str, prompt: str, json_mode: bool = False
    ) -> Optional[str]:
        chain = self._config.get_role_chain(role)
        if not chain:
            return f"ERROR: No models defined for role '{role}' in YAML."

        errors: list[str] = []
        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider")
            logger.info(f"[{role.upper()}] Routing to: {model_id} ({provider_name})")

            try:
                # Get provider from cache or create new
                if model_id in self._provider_cache:
                    provider = self._provider_cache[model_id]
                else:
                    provider = get_provider(provider_name, model_cfg)
                    self._provider_cache[model_id] = provider
                    
                result = provider.generate(prompt, json_mode=json_mode)

                if result and result.strip():
                    logger.info(f"[{role.upper()}] Success from {model_id}.")
                    return result
                else:
                    raise ValueError("Response was empty or invalid.")

            except Exception as e:
                err = f"{model_id} failed: {e}"
                logger.warning(err)
                errors.append(err)
                continue

        logger.critical(
            f"All nodes failed for role '{role}'. Errors: {errors}"
        )
        return None
