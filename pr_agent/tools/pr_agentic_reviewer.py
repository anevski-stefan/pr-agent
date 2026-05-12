import asyncio
import datetime
from dataclasses import asdict, dataclass, field

import yaml
from jinja2 import Environment, StrictUndefined

from pr_agent.algo.pr_processing import get_pr_diff
from pr_agent.algo.skill_loader import SkillDefinition, get_skill_descriptions_for_triage, load_review_skills
from pr_agent.algo.utils import load_yaml
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

_SKILL_REVIEW_WRAPPER = """\
You are acting as a code reviewer using the following expertise and guidelines.
Review ONLY the new code in the diff (lines starting with '+') for issues relevant to your expertise.
Do NOT suggest implementing new features or refactoring — only flag real problems introduced by this PR.

Output a YAML object:
```yaml
findings:
  - relevant_file: path/to/file.py
    issue_header: Short Title
    issue_content: Concrete description and realistic trigger scenario.
    start_line: 10
    end_line: 14
    severity: critical|high|medium|low
```

Output an empty list if no relevant issues are found:
```yaml
findings: []
```

--- YOUR EXPERTISE AND GUIDELINES ---
"""


@dataclass
class TriageResult:
    selected_skills: list[str] = field(default_factory=list)
    file_risk_scores: list[dict] = field(default_factory=list)
    initial_findings: list[dict] = field(default_factory=list)


@dataclass
class SkillFinding:
    relevant_file: str
    issue_header: str
    issue_content: str
    start_line: int
    end_line: int
    skill: str
    severity: str = "medium"


class AgenticPRReviewer:
    def __init__(self, reviewer):
        self.reviewer = reviewer
        self.git_provider = reviewer.git_provider
        self.ai_handler = reviewer.ai_handler
        self.vars = reviewer.vars

    # ------------------------------------------------------------------ #
    # Skill discovery                                                    #
    # ------------------------------------------------------------------ #

    def _discover_skills(self) -> list[SkillDefinition]:
        enabled_names = list(get_settings().pr_reviewer_agent.agent_review_skills)
        branch = self.vars.get("branch", "")
        return load_review_skills(enabled_names, self.git_provider, branch)

    # ------------------------------------------------------------------ #
    # Triage phase                                                       #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Skill execution phase                                              #
    # ------------------------------------------------------------------ #

    def _get_skill_model(self) -> str:
        skill_model = get_settings().pr_reviewer_agent.get("agent_skill_call_model", "")
        return skill_model if skill_model else get_settings().config.model

    def _fetch_file_contexts(self, files: list[str]) -> dict[str, str]:
        branch = self.vars.get("branch", "")
        contexts: dict[str, str] = {}
        for file in files:
            try:
                content = self.git_provider.get_pr_file_content(file, branch)
                if content:
                    contexts[file] = content
            except Exception as e:
                get_logger().warning(f"Could not fetch content for '{file}': {e}")
        return contexts

    def _build_skill_system_prompt(self, skill: SkillDefinition) -> str:
        return _SKILL_REVIEW_WRAPPER + skill.content

    def _build_skill_user_prompt(self, diff: str, file_contexts: dict[str, str]) -> str:
        parts = []
        if file_contexts:
            parts.append("Full file contents for high-risk files:")
            for path, content in file_contexts.items():
                parts.append(f"\n## File: '{path}'\n{content}")
            parts.append("\n")
        parts.append(f"PR diff:\n======\n{diff}\n======")
        return "\n".join(parts)

    def _parse_skill_response(self, response: str, skill_name: str) -> list[SkillFinding]:
        data = load_yaml(response.strip(), first_key="findings")
        if not data or "findings" not in data:
            get_logger().warning(f"Skill '{skill_name}' returned no parseable findings")
            return []

        findings = []
        for f in (data.get("findings") or []):
            if not isinstance(f, dict):
                continue
            try:
                findings.append(SkillFinding(
                    relevant_file=f.get("relevant_file", ""),
                    issue_header=f.get("issue_header", ""),
                    issue_content=f.get("issue_content", ""),
                    start_line=int(f.get("start_line", 0)),
                    end_line=int(f.get("end_line", 0)),
                    severity=f.get("severity", "medium"),
                    skill=skill_name,
                ))
            except Exception as e:
                get_logger().warning(f"Failed to parse finding from skill '{skill_name}': {e}")

        max_per_skill = get_settings().pr_reviewer_agent.agent_max_findings_per_skill
        return findings[:max_per_skill]

    async def _run_skill(self, skill: SkillDefinition, model: str, diff: str, file_contexts: dict[str, str]) -> list[SkillFinding]:
        system = self._build_skill_system_prompt(skill)
        user = self._build_skill_user_prompt(diff, file_contexts)
        response, _ = await self.ai_handler.chat_completion(
            model=model,
            temperature=get_settings().config.temperature,
            system=system,
            user=user,
        )
        return self._parse_skill_response(response, skill.name)

    async def _run_skills_parallel(self, triggered_skills: list[SkillDefinition], triage: TriageResult) -> list[SkillFinding]:
        if not triggered_skills:
            return []

        max_calls = get_settings().pr_reviewer_agent.agent_max_skill_calls
        skills_to_run = triggered_skills[:max_calls]
        model = self._get_skill_model()

        min_risk = get_settings().pr_reviewer_agent.agent_min_risk_to_trigger
        high_risk_files = [f["file"] for f in triage.file_risk_scores if f.get("risk", 0) >= min_risk]
        file_contexts = self._fetch_file_contexts(high_risk_files)
        diff = self.reviewer.patches_diff or ""

        coros = [self._run_skill(skill, model, diff, file_contexts) for skill in skills_to_run]

        if get_settings().pr_reviewer_agent.get("agent_parallel_skills", True):
            raw_results = await asyncio.gather(*coros, return_exceptions=True)
        else:
            raw_results = []
            for coro in coros:
                try:
                    raw_results.append(await coro)
                except Exception as e:
                    raw_results.append(e)

        findings: list[SkillFinding] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                get_logger().error(f"Skill '{skills_to_run[i].name}' failed: {result}")
            else:
                findings.extend(result)
        return findings

    # ------------------------------------------------------------------ #
    # Merge phase                                                        #
    # ------------------------------------------------------------------ #

    _SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def _merge_findings(self, triage_findings: list[dict], skill_findings: list[SkillFinding]) -> list[dict]:
        skill_dicts = [asdict(f) for f in skill_findings]

        merged = list(triage_findings)
        for skill_finding in skill_dicts:
            duplicate_idx = next(
                (i for i, e in enumerate(merged)
                 if e.get("relevant_file") == skill_finding["relevant_file"]
                 and abs(e.get("start_line", 0) - skill_finding["start_line"]) <= 3),
                None,
            )
            if duplicate_idx is not None:
                merged[duplicate_idx] = skill_finding  
            else:
                merged.append(skill_finding)

        merged.sort(key=lambda f: self._SEVERITY_ORDER.get(f.get("severity", "medium"), 2))

        num_max = get_settings().pr_reviewer.num_max_findings
        return merged[:num_max]

    # ------------------------------------------------------------------ #
    # Publishing                                                         #
    # ------------------------------------------------------------------ #

    def _format_as_prediction(self, findings: list[dict]) -> str:
        """Convert findings into the YAML format PRReviewer._prepare_pr_review() expects."""
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

        triggered_skills = [s for s in skills if s.name in triage.selected_skills]
        skill_findings = await self._run_skills_parallel(triggered_skills, triage)

        merged = self._merge_findings(triage.initial_findings, skill_findings)

        self.reviewer.prediction = self._format_as_prediction(merged)
        pr_review = self.reviewer._prepare_pr_review()

        if pr_review and get_settings().config.publish_output:
            self.reviewer.git_provider.publish_comment(pr_review)
            self.reviewer.git_provider.remove_initial_comment()
