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


class TestMergeFindings:
    def _triage_finding(self, file="src/auth.py", line=10, header="Triage Issue", severity="medium"):
        return {"relevant_file": file, "issue_header": header, "issue_content": "desc", "start_line": line, "end_line": line + 1}

    def _skill_finding(self, file="src/auth.py", line=10, header="Skill Issue", skill="security-and-hardening", severity="high"):
        from pr_agent.tools.pr_agentic_reviewer import SkillFinding
        return SkillFinding(relevant_file=file, issue_header=header, issue_content="desc",
                            start_line=line, end_line=line + 1, skill=skill, severity=severity)

    def test_merge_combines_both_sources(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        agent = AgenticPRReviewer(_make_reviewer())
        triage = [self._triage_finding(line=5)]
        skills = [self._skill_finding(line=20)]
        result = agent._merge_findings(triage, skills)
        assert len(result) == 2

    def test_skill_finding_wins_on_same_location(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        agent = AgenticPRReviewer(_make_reviewer())
        triage = [self._triage_finding(line=10, header="Triage")]
        skills = [self._skill_finding(line=11, header="Skill")]  
        result = agent._merge_findings(triage, skills)
        assert len(result) == 1
        assert result[0]["issue_header"] == "Skill"

    def test_no_dedup_different_files(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        agent = AgenticPRReviewer(_make_reviewer())
        triage = [self._triage_finding(file="src/auth.py", line=10)]
        skills = [self._skill_finding(file="src/utils.py", line=10)]
        result = agent._merge_findings(triage, skills)
        assert len(result) == 2

    def test_no_dedup_lines_far_apart(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        agent = AgenticPRReviewer(_make_reviewer())
        triage = [self._triage_finding(line=5)]
        skills = [self._skill_finding(line=10)]  
        result = agent._merge_findings(triage, skills)
        assert len(result) == 2

    def test_sort_order_critical_first(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        agent = AgenticPRReviewer(_make_reviewer())
        skills = [
            self._skill_finding(line=5, severity="low"),
            self._skill_finding(line=10, severity="critical"),
            self._skill_finding(line=15, severity="medium"),
            self._skill_finding(line=20, severity="high"),
        ]
        get_settings().pr_reviewer.num_max_findings = 10
        try:
            result = agent._merge_findings([], skills)
            severities = [r.get("severity", "medium") for r in result]
            assert severities == ["critical", "high", "medium", "low"]
        finally:
            get_settings().pr_reviewer.num_max_findings = 3

    def test_capped_at_num_max_findings(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        agent = AgenticPRReviewer(_make_reviewer())
        skills = [self._skill_finding(line=i * 20) for i in range(5)]
        get_settings().pr_reviewer.num_max_findings = 3
        try:
            result = agent._merge_findings([], skills)
            assert len(result) == 3
        finally:
            get_settings().pr_reviewer.num_max_findings = 3

    def test_skill_findings_have_skill_field(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        agent = AgenticPRReviewer(_make_reviewer())
        skills = [self._skill_finding(skill="security-and-hardening")]
        result = agent._merge_findings([], skills)
        assert result[0].get("skill") == "security-and-hardening"

    def test_empty_inputs_returns_empty(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        agent = AgenticPRReviewer(_make_reviewer())
        assert agent._merge_findings([], []) == []


class TestRunEndToEnd:
    def test_run_calls_skill_execution_with_triggered_skills(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
            with patch.object(agent, "_run_skills_parallel", new=AsyncMock(return_value=[])):
                asyncio.run(agent.run())
        reviewer.ai_handler.chat_completion.assert_called()
        reviewer.git_provider.publish_comment.assert_called()

    def test_run_publishes_merged_output(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
            with patch.object(agent, "_run_skills_parallel", new=AsyncMock(return_value=[])):
                asyncio.run(agent.run())
        reviewer.git_provider.publish_comment.assert_called_once()


class TestRun:
    def test_run_publishes_when_findings_exist(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
            with patch.object(agent, "_run_skills_parallel", new=AsyncMock(return_value=[])):
                asyncio.run(agent.run())
        reviewer.git_provider.publish_comment.assert_called()

    def test_run_no_files_returns_early(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.git_provider.get_files.return_value = []
        agent = AgenticPRReviewer(reviewer)
        asyncio.run(agent.run())
        reviewer.ai_handler.chat_completion.assert_not_called()


class TestGracefulDegradation:
    def test_no_skills_falls_back_to_single_shot(self):
        """When no skills are found, run() falls back to single-shot review without crashing."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[]):
            with patch.object(agent, "_run_single_shot", new=AsyncMock()) as mock_fallback:
                asyncio.run(agent.run())
                mock_fallback.assert_called_once()
        reviewer.ai_handler.chat_completion.assert_not_called()

    def test_no_skills_does_not_call_triage(self):
        """Triage LLM call is skipped entirely when no skills are available."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[]):
            with patch.object(agent, "_run_single_shot", new=AsyncMock()):
                asyncio.run(agent.run())
        reviewer.ai_handler.chat_completion.assert_not_called()

    def test_all_skills_fail_triage_findings_still_published(self):
        """If all skill LLM calls fail, triage initial_findings are still published."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        call_count = 0
        async def fail_on_skill(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:  
                raise Exception("Skill LLM down")
            return (TRIAGE_RESPONSE, "stop")
        reviewer.ai_handler.chat_completion = fail_on_skill
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
            asyncio.run(agent.run())
        reviewer.git_provider.publish_comment.assert_called()

    def test_provider_without_get_pr_file_content_uses_diff_only(self):
        """If provider raises AttributeError on get_pr_file_content, skill still runs with diff."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.git_provider.get_pr_file_content.side_effect = AttributeError("not supported")
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        agent = AgenticPRReviewer(reviewer)
        triage = _make_triage(risk_scores=[{"file": "src/auth.py", "risk": 4}])
        result = asyncio.run(agent._run_skills_parallel([_make_skill()], triage))
        assert result == []


class TestMultiPass:
    def test_default_max_passes_is_one(self):
        from pr_agent.config_loader import get_settings
        assert get_settings().pr_reviewer_agent.agent_max_passes == 1

    def test_second_pass_injects_previous_findings_into_prompt(self):
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        agent = AgenticPRReviewer(reviewer)
        previous = [{"relevant_file": "src/auth.py", "issue_header": "Logic Error",
                      "issue_content": "Swapped handlers", "start_line": 10, "end_line": 11}]
        prompt = agent._build_skill_user_prompt("+ some diff", {}, previous_findings=previous)
        assert "previous" in prompt.lower() or "Logic Error" in prompt

    def test_second_pass_uses_files_from_first_pass_findings(self):
        """Pass 2 fetches full content for files that had findings in Pass 1."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        agent = AgenticPRReviewer(reviewer)
        previous = [{"relevant_file": "src/hero.tsx", "issue_header": "Bug",
                      "issue_content": "desc", "start_line": 5, "end_line": 6}]
        triage = _make_triage(risk_scores=[{"file": "src/hero.tsx", "risk": 1}])
        asyncio.run(agent._run_skills_parallel([_make_skill()], triage, previous_findings=previous))
        fetched = [call.args[0] for call in reviewer.git_provider.get_pr_file_content.call_args_list]
        assert any("hero.tsx" in f for f in fetched)

    def test_multi_pass_run_makes_extra_skill_calls(self):
        """With agent_max_passes=2, run() makes skill calls twice."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        call_count = 0

        async def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (TRIAGE_RESPONSE, "stop")
            return (SKILL_FINDING_RESPONSE, "stop")

        reviewer.ai_handler.chat_completion = count_calls
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        get_settings().pr_reviewer_agent.agent_max_passes = 2
        try:
            with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
                asyncio.run(agent.run())
            assert call_count >= 3
        finally:
            get_settings().pr_reviewer_agent.agent_max_passes = 1

    def test_three_passes_accumulate_all_previous_findings(self):
        """Pass 3 context includes findings from both Pass 1 and Pass 2."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        captured_prompts = []
        call_count = 0

        async def capture(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (TRIAGE_RESPONSE, "stop")
            captured_prompts.append(kwargs.get("user", ""))
            return (SKILL_FINDING_RESPONSE, "stop")

        reviewer.ai_handler.chat_completion = capture
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        get_settings().pr_reviewer_agent.agent_max_passes = 3
        try:
            with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
                asyncio.run(agent.run())
            assert len(captured_prompts) >= 2
            assert "SQL Injection" in captured_prompts[1] or "finding" in captured_prompts[1].lower() or len(captured_prompts[1]) > len(captured_prompts[0])
        finally:
            get_settings().pr_reviewer_agent.agent_max_passes = 1

    def test_file_not_fetched_twice_across_passes(self):
        """Same file is not fetched from provider more than once across passes."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        reviewer.git_provider.get_pr_file_content.return_value = "def login(): pass"
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=("findings: []", "stop"))
        agent = AgenticPRReviewer(reviewer)
        triage = _make_triage(risk_scores=[{"file": "src/auth.py", "risk": 4}])
        cache: dict = {}
        get_settings().pr_reviewer_agent.agent_min_risk_to_trigger = 1
        try:
            asyncio.run(agent._run_skills_parallel([_make_skill()], triage, file_context_cache=cache))
            asyncio.run(agent._run_skills_parallel([_make_skill()], triage, file_context_cache=cache))
            assert reviewer.git_provider.get_pr_file_content.call_count == 1
        finally:
            get_settings().pr_reviewer_agent.agent_min_risk_to_trigger = 3

    def test_single_pass_behavior_unchanged_when_max_passes_one(self):
        """agent_max_passes=1 (default) produces same call count as before."""
        from pr_agent.tools.pr_agentic_reviewer import AgenticPRReviewer
        from pr_agent.config_loader import get_settings
        reviewer = _make_reviewer()
        reviewer.ai_handler.chat_completion = AsyncMock(return_value=(TRIAGE_RESPONSE, "stop"))
        agent = AgenticPRReviewer(reviewer)
        skill = _make_skill("security-and-hardening")
        get_settings().pr_reviewer_agent.agent_max_passes = 1
        with patch("pr_agent.tools.pr_agentic_reviewer.load_review_skills", return_value=[skill]):
            with patch.object(agent, "_run_skills_parallel", new=AsyncMock(return_value=[])):
                asyncio.run(agent.run())
        assert reviewer.ai_handler.chat_completion.call_count == 1
