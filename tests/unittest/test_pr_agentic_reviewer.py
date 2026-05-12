import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pr_agent.algo.skill_loader import SkillDefinition

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRIAGE_RESPONSE = (FIXTURES / "triage_response.yaml").read_text()
SKILL_FINDING_RESPONSE = (FIXTURES / "skill_finding_response.yaml").read_text()
SECURITY_MD = (FIXTURES / "skills" / "security_skill.md").read_text()


def _make_reviewer(patches_diff="+ some diff"):
    """Build a minimal PRReviewer-shaped mock."""
    reviewer = MagicMock()
    reviewer.git_provider.get_files.return_value = ["src/auth.py"]
    reviewer.git_provider.get_pr_branch.return_value = "feature/login"
    reviewer.git_provider.get_pr_file_content.return_value = ""
    reviewer.vars = {
        "title": "Add login",
        "branch": "feature/login",
        "description": "Adds login endpoint",
        "date": "2026-05-12",
        "diff": patches_diff,
    }
    reviewer.patches_diff = patches_diff
    reviewer.prediction = None
    reviewer.ai_handler = MagicMock()
    reviewer.ai_handler.chat_completion = AsyncMock(
        return_value=(TRIAGE_RESPONSE, "stop")
    )
    reviewer.token_handler = MagicMock()
    return reviewer


class TestAgenticPRReviewerInit:
    def test_init_inherits_git_provider(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        assert agent.git_provider is reviewer.git_provider

    def test_init_inherits_ai_handler(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        assert agent.ai_handler is reviewer.ai_handler

    def test_init_inherits_vars(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        assert agent.vars is reviewer.vars


class TestDiscoverSkills:
    def test_discover_skills_calls_load_review_skills(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills") as mock_load:
            mock_load.return_value = []
            agent._discover_skills()
            mock_load.assert_called_once()

    def test_discover_skills_passes_branch_from_vars(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills") as mock_load:
            mock_load.return_value = []
            agent._discover_skills()
            args = mock_load.call_args[0]
            assert "feature/login" in args

    def test_discover_skills_returns_skill_definitions(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        skill = SkillDefinition(name="security-and-hardening", description="Security.", content=SECURITY_MD)
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
            result = agent._discover_skills()
        assert len(result) == 1
        assert result[0].name == "security-and-hardening"


class TestRunTriage:
    def test_triage_calls_chat_completion_once(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        asyncio.run(agent._run_triage([]))
        reviewer.ai_handler.chat_completion.assert_called_once()

    def test_triage_uses_temperature_zero(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        asyncio.run(agent._run_triage([]))
        _, kwargs = reviewer.ai_handler.chat_completion.call_args
        assert kwargs.get("temperature") == 0

    def test_triage_uses_config_model_when_triage_model_empty(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        asyncio.run(agent._run_triage([]))
        _, kwargs = reviewer.ai_handler.chat_completion.call_args
        assert kwargs.get("model") == get_settings().config.model

    def test_triage_uses_agent_triage_model_when_set(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        get_settings().pr_reviewer_agent.agent_triage_model = "openrouter/openai/gpt-4o-mini"
        try:
            asyncio.run(agent._run_triage([]))
            _, kwargs = reviewer.ai_handler.chat_completion.call_args
            assert kwargs.get("model") == "openrouter/openai/gpt-4o-mini"
        finally:
            get_settings().pr_reviewer_agent.agent_triage_model = ""

    def test_triage_parses_selected_skills(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_triage([]))
        assert "security-and-hardening" in result.selected_skills
        assert "code-review-and-quality" in result.selected_skills

    def test_triage_parses_file_risk_scores(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_triage([]))
        assert len(result.file_risk_scores) == 2
        assert result.file_risk_scores[0]["file"] == "src/auth.py"
        assert result.file_risk_scores[0]["risk"] == 4

    def test_triage_parses_initial_findings(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_triage([]))
        assert len(result.initial_findings) == 1
        assert result.initial_findings[0]["issue_header"] == "Missing Rate Limit"

    def test_triage_invalid_yaml_returns_empty_result(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("not valid yaml {{{{", "stop"))
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_triage([]))
        assert result.selected_skills == []
        assert result.initial_findings == []

    def test_triage_missing_triage_key_returns_empty_result(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("review:\n  key: value", "stop"))
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_triage([]))
        assert result.selected_skills == []
        assert result.initial_findings == []


def _make_skill(name="security-and-hardening"):
    return SkillDefinition(name=name, description="Security checks.", content=SECURITY_MD)


def _make_triage(selected=("security-and-hardening",), risk_scores=None):
    from pr_agent.tools.pr_agentic_reviewer import TriageResult
    return TriageResult(
        selected_skills=list(selected),
        file_risk_scores=risk_scores or [{"file": "src/auth.py", "risk": 4}],
        initial_findings=[],
    )


class TestRunSkillsParallel:
    def test_empty_skills_returns_empty_list(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_skills_parallel([], _make_triage(selected=[])))
        assert result == []
        reviewer.ai_handler.chat_completion.assert_not_called()

    def test_hard_cap_limits_skill_calls(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=(SKILL_FINDING_RESPONSE, "stop"))
        agent = AgenticPRReviewer(reviewer)
        skills = [_make_skill(f"skill-{i}") for i in range(5)]
        triage = _make_triage(selected=[s.name for s in skills])
        get_settings().pr_reviewer_agent.agent_max_skill_calls = 3
        try:
            asyncio.run(agent._run_skills_parallel(skills, triage))
            assert reviewer.ai_handler.chat_completion.call_count == 3
        finally:
            get_settings().pr_reviewer_agent.agent_max_skill_calls = 3

    def test_findings_include_skill_name(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=(SKILL_FINDING_RESPONSE, "stop"))
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        result = asyncio.run(agent._run_skills_parallel([skill], _make_triage()))
        assert all(f.skill == "security-and-hardening" for f in result)

    def test_high_risk_files_fetched_from_provider(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        reviewer.git_provider.get_pr_file_content.return_value = "def login(): pass"
        agent = AgenticPRReviewer(reviewer)
        triage = _make_triage(risk_scores=[{"file": "src/auth.py", "risk": 4}])
        get_settings().pr_reviewer_agent.agent_min_risk_to_trigger = 3
        asyncio.run(agent._run_skills_parallel([_make_skill()], triage))
        reviewer.git_provider.get_pr_file_content.assert_called()

    def test_low_risk_files_not_fetched(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        agent = AgenticPRReviewer(reviewer)
        triage = _make_triage(risk_scores=[{"file": "src/utils.py", "risk": 1}])
        get_settings().pr_reviewer_agent.agent_min_risk_to_trigger = 3
        asyncio.run(agent._run_skills_parallel([_make_skill()], triage))
        reviewer.git_provider.get_pr_file_content.assert_not_called()

    def test_get_pr_file_content_exception_does_not_crash(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.git_provider.get_pr_file_content.side_effect = Exception("Network error")
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        agent = AgenticPRReviewer(reviewer)
        result = asyncio.run(agent._run_skills_parallel([_make_skill()], _make_triage()))
        assert result == []

    def test_one_skill_exception_does_not_cancel_others(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM error")
            return (SKILL_FINDING_RESPONSE, "stop")

        reviewer.ai_handler.chat_completion = flaky
        agent = AgenticPRReviewer(reviewer)
        skills = [_make_skill("skill-a"), _make_skill("skill-b")]
        result = asyncio.run(agent._run_skills_parallel(skills, _make_triage(selected=["skill-a", "skill-b"])))
        assert len(result) > 0

    def test_sequential_mode_exception_does_not_cancel_others(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("LLM error on first skill")
            return (SKILL_FINDING_RESPONSE, "stop")

        reviewer.ai_handler.chat_completion = flaky
        agent = AgenticPRReviewer(reviewer)
        get_settings().pr_reviewer_agent.agent_parallel_skills = False
        try:
            skills = [_make_skill("skill-a"), _make_skill("skill-b")]
            result = asyncio.run(agent._run_skills_parallel(skills, _make_triage(selected=["skill-a", "skill-b"])))
            assert len(result) > 0
        finally:
            get_settings().pr_reviewer_agent.agent_parallel_skills = True

    def test_sequential_mode_respected(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        agent = AgenticPRReviewer(reviewer)
        get_settings().pr_reviewer_agent.agent_parallel_skills = False
        try:
            skills = [_make_skill("skill-a"), _make_skill("skill-b")]
            asyncio.run(agent._run_skills_parallel(skills, _make_triage(selected=["skill-a", "skill-b"])))
            assert reviewer.ai_handler.chat_completion.call_count == 2
        finally:
            get_settings().pr_reviewer_agent.agent_parallel_skills = True

    def test_per_skill_findings_capped(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings

        big_response = "findings:\n" + "  - relevant_file: f.py\n    issue_header: Issue\n    issue_content: Desc.\n    start_line: 1\n    end_line: 1\n    severity: high\n" * 3
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=(big_response, "stop"))
        agent = AgenticPRReviewer(reviewer)
        get_settings().pr_reviewer_agent.agent_max_findings_per_skill = 2
        try:
            result = asyncio.run(agent._run_skills_parallel([_make_skill()], _make_triage()))
            assert len(result) <= 2
        finally:
            get_settings().pr_reviewer_agent.agent_max_findings_per_skill = 5


class TestRun:
    def test_run_publishes_when_findings_exist(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[]):
            asyncio.run(agent.run())
        reviewer.git_provider.publish_comment.assert_called()

    def test_run_no_files_returns_early(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.git_provider.get_files.return_value = []
        agent = AgenticPRReviewer(reviewer)
        asyncio.run(agent.run())
        reviewer.ai_handler.chat_completion.assert_not_called()
