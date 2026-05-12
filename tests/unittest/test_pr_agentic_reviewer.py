import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pr_agent.algo.skill_loader import SkillDefinition

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRIAGE_RESPONSE = (FIXTURES / "triage_response.yaml").read_text()
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
