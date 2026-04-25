"""
AdversarialExtractionEnv — OpenEnv compliant two-agent environment.

Episode flow (single step per episode, done=True after extractor acts):
  1. reset() → sample document + schema, return (ExtractorObs, AdversaryObs)
  2. step_adversary(AdversaryAction) → apply edits, return AdversaryObs
  3. step_extractor(ExtractorAction) → score, return (ExtractorObs, reward, done=True, info)
"""

import uuid
from typing import Tuple, Optional, Any, Union
from data.corpus import DocumentCorpus
from env.models import *
from env.adversary import AdversaryEditExecutor
from env.rubric import build_extractor_rubric, build_adversary_rubric
from openenv.core.env_server.interfaces import Environment

class EnvAction(BaseModel):
    # Wrapper to support multiple action types if needed, though step will accept Union directly
    action: Union[ExtractorAction, AdversaryAction]

class EnvObservation(BaseModel):
    observation: Union[ExtractionObservation, AdversaryObservation]

class AdversarialExtractionEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    PERTURBATION_BUDGET_DEFAULT = 200

    def __init__(
        self,
        split: str = "train",
        token_budget_mode: str = "linear",
        perturbation_budget: int = 200,
        adversary_policy: str = "random",   # "random" | "model" | "none"
    ):
        super().__init__()
        self.corpus = DocumentCorpus(split=split)
        self.extractor_rubric = build_extractor_rubric(token_budget_mode)
        self.adversary_rubric = build_adversary_rubric()
        self.executor = AdversaryEditExecutor()
        self.perturbation_budget = perturbation_budget
        self.adversary_policy = adversary_policy
        self._state: Optional[EpisodeState] = None
        self._extractor_reward_history: list = []

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> EnvObservation:
        doc = self.corpus.sample()
        self._state = EpisodeState(
            episode_id=str(uuid.uuid4()) if not episode_id else episode_id,
            document_original=doc["text"],
            document_current=doc["text"],
            schema=doc["schema"],
            gold_answers=doc["gold"],
            doc_type=doc["type"],
        )
        extractor_obs = ExtractionObservation(
            document_text=self._state.document_current,
            target_schema=self._state.schema,
            token_budget_remaining=512,
            step=0,
            metadata={"episode_id": self._state.episode_id, "doc_type": doc["type"],
                       "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        )
        adversary_obs = AdversaryObservation(
            document_text=self._state.document_original,
            target_schema=self._state.schema,
            extractor_reward_history=list(self._extractor_reward_history[-10:]),
            token_budget_remaining=self.perturbation_budget,
            perturbation_budget=self.perturbation_budget,
            step=0,
        )
        # If adversary is a model and acts first, we return adversary_obs
        # For simplicity, returning a combined or current turn observation.
        return EnvObservation(observation=adversary_obs if self.adversary_policy == "model" else extractor_obs)

    def step(self, action: EnvAction, timeout_s: Optional[float] = None, **kwargs: Any) -> EnvObservation:
        if isinstance(action.action, AdversaryAction):
            return EnvObservation(observation=self.step_adversary(action.action))
        else:
            obs, reward, done, info = self.step_extractor(action.action)
            # In OpenEnv, step returns Observation. The reward is added to the obs or state.
            obs.reward = reward
            return EnvObservation(observation=obs)

    @property
    def state(self) -> Any:
        return self._state

    def step_adversary(self, action: AdversaryAction) -> ExtractionObservation:
        """Apply adversary edits. Returns updated extractor observation."""
        assert self._state is not None
        if not self.executor.validate_budget(action.edits, self.perturbation_budget):
            action.edits = []
        modified_doc, modified_schema = self.executor.apply_edits(
            self._state.document_current, self._state.schema, action.edits
        )
        self._state.document_current = modified_doc
        self._state.schema = modified_schema
        self._state.adversary_action = action
        self._state.applied_edits = action.edits
        perturbation_log = [f"{e.edit_type.value}({e.params})" for e in action.edits]
        return ExtractionObservation(
            document_text=modified_doc,
            target_schema=modified_schema,
            perturbation_history=perturbation_log,
            token_budget_remaining=512,
            step=1,
            metadata={"episode_id": self._state.episode_id, "doc_type": self._state.doc_type,
                       "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        )

    def step_extractor(self, action: ExtractorAction) -> Tuple[ExtractionObservation, float, bool, dict]:
        """Score extraction. Returns (obs, reward, done=True, info)."""
        assert self._state is not None
        self._state.extractor_action = action
        
        if self._extractor_reward_history:
            self._state.baseline_extractor_reward = sum(self._extractor_reward_history[-10:]) / min(10, len(self._extractor_reward_history))
        else:
            self._state.baseline_extractor_reward = 0.5
            
        extractor_reward = self.extractor_rubric(self._state)
        adversary_reward = self.adversary_rubric(self._state)
        
        self._state.extractor_reward = extractor_reward
        self._state.adversary_reward = adversary_reward
        self._state.done = True
        self._extractor_reward_history.append(extractor_reward)
        
        obs = ExtractionObservation(
            document_text=self._state.document_current,
            target_schema=self._state.schema,
            token_budget_remaining=0,
            step=2,
            reward=extractor_reward,
        )
        info = {
            "extractor_reward": extractor_reward,
            "adversary_reward": adversary_reward,
            "doc_type": self._state.doc_type,
            "edits_applied": len(self._state.applied_edits),
            "field_scores": {}, 
        }
        return obs, extractor_reward, True, info

    def render(self) -> str:
        if not self._state:
            return "No active episode."
        return (
            f"Episode: {self._state.episode_id[:8]}\n"
            f"Doc type: {self._state.doc_type}\n"
            f"Edits applied: {len(self._state.applied_edits)}\n"
            f"Extractor reward: {self._state.extractor_reward:.3f}\n"
            f"Adversary reward: {self._state.adversary_reward:.3f}\n"
        )
