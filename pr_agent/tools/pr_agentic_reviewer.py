import datetime
from dataclasses import dataclass, field

import yaml
from jinja2 import Environment, StrictUndefined

from pr_agent.algo.pr_processing import get_pr_diff
from pr_agent.algo.skill_loader import SkillDefinition, get_skill_descriptions_for_triage, load_review_skills
from pr_agent.algo.utils import load_yaml
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger


@dataclass
class TriageResult:
    selected_skills: list[str] = field(default_factory=list)
    file_risk_scores: list[dict] = field(default_factory=list)
    initial_findings: list[dict] = field(default_factory=list)


class AgenticPRReviewer:
    def __init__(self, reviewer):
        self.reviewer = reviewer
        self.git_provider = reviewer.git_provider
        self.ai_handler = reviewer.ai_handler
        self.vars = reviewer.vars

    def _discover_skills(self) -> list[SkillDefinition]:
        enabled_names = list(get_settings().pr_reviewer_agent.agent_review_skills)
        branch = self.vars.get("branch", "")
        return load_review_skills(enabled_names, self.git_provider, branch)

    def _get_triage_model(self) -> str:
        triage_model = get_settings().pr_reviewer_agent.get("agent_triage_model", "")
        return triage_model if triage_model else get_settings().config.model

    def _parse_triage_response(self, response: str) -> TriageResult:
        data = load_yaml(response.strip(), first_key="triage")
        if not data or "triage" not in data:
            get_logger().warning("Failed to parse triage response, using empty result")
            return TriageResult()

        triage = data["triage"]
        return TriageResult(
            selected_skills=triage.get("skill_triggers") or [],
            file_risk_scores=triage.get("file_risk_scores") or [],
            initial_findings=triage.get("initial_findings") or [],
        )

    async def _run_triage(self, skills: list[SkillDefinition]) -> TriageResult:
        model = self._get_triage_model()

        diff = self.reviewer.patches_diff or ""
        if not diff and hasattr(self.reviewer, "token_handler"):
            diff = get_pr_diff(
                self.git_provider,
                self.reviewer.token_handler,
                model,
                add_line_numbers_to_hunks=True,
            ) or ""

        variables = dict(self.vars)
        variables["diff"] = diff
        variables["skill_descriptions"] = get_skill_descriptions_for_triage(skills)
        variables["agent_max_skill_calls"] = get_settings().pr_reviewer_agent.agent_max_skill_calls
        variables.setdefault("date", datetime.datetime.now().strftime("%Y-%m-%d"))

        env = Environment(undefined=StrictUndefined)
        system = get_settings().pr_reviewer_triage_prompt.system
        user = env.from_string(get_settings().pr_reviewer_triage_prompt.user).render(variables)

        response, _ = await self.ai_handler.chat_completion(
            model=model,
            temperature=0,
            system=system,
            user=user,
        )
        return self._parse_triage_response(response)

    def _format_as_prediction(self, findings: list[dict]) -> str:
        """Convert triage findings into the YAML format PRReviewer._prepare_pr_review() expects."""
        review: dict = {"key_issues_to_review": findings or []}
        if get_settings().pr_reviewer.require_estimate_effort_to_review:
            review["estimated_effort_to_review_[1-5]"] = 1
        if get_settings().pr_reviewer.require_tests_review:
            review["relevant_tests"] = "No"
        if get_settings().pr_reviewer.require_security_review:
            review["security_concerns"] = "No"
        return yaml.dump({"review": review}, allow_unicode=True, default_flow_style=False)

    async def run(self) -> None:
        if not self.git_provider.get_files():
            get_logger().info("PR has no files, skipping agentic review")
            return

        skills = self._discover_skills()
        triage = await self._run_triage(skills)

        self.reviewer.prediction = self._format_as_prediction(triage.initial_findings)
        pr_review = self.reviewer._prepare_pr_review()

        if pr_review and get_settings().config.publish_output:
            self.reviewer.git_provider.publish_comment(pr_review)
            self.reviewer.git_provider.remove_initial_comment()
