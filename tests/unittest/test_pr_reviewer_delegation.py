import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _make_pr_reviewer(agent_mode=False):
    """Instantiate PRReviewer with a mocked git provider and ai_handler."""
    from pr_agent.config_loader import get_settings
    get_settings().pr_reviewer_agent.agent_mode = agent_mode

    with patch("pr_agent.tools.pr_reviewer.get_git_provider_with_context") as mock_provider:
        mock_git = MagicMock()
        mock_git.get_files.return_value = ["src/auth.py"]
        mock_git.get_pr_branch.return_value = "main"
        mock_git.get_languages.return_value = {"Python": 100}
        mock_git.get_num_of_files.return_value = 1
        mock_git.get_commit_messages.return_value = ""
        mock_git.pr.title = "Test PR"
        mock_git.get_pr_description.return_value = ("desc", None)
        mock_provider.return_value = mock_git

        from pr_agent.tools.pr_reviewer import PRReviewer

        reviewer = PRReviewer.__new__(PRReviewer)
        reviewer.git_provider = mock_git
        reviewer.pr_url = "https://github.com/org/repo/pull/1"
        reviewer.is_answer = False
        reviewer.is_auto = False
        reviewer.args = None
        reviewer.patches_diff = None
        reviewer.prediction = None
        reviewer.vars = {
            "title": "Test PR",
            "branch": "main",
            "description": "desc",
            "language": "Python",
            "diff": "",
            "num_pr_files": 1,
            "num_max_findings": 3,
            "require_score": False,
            "require_tests": True,
            "require_estimate_effort_to_review": True,
            "require_estimate_contribution_time_cost": False,
            "require_can_be_split_review": False,
            "require_security_review": True,
            "require_todo_scan": False,
            "question_str": "",
            "answer_str": "",
            "extra_instructions": "",
            "commit_messages_str": "",
            "custom_labels": "",
            "enable_custom_labels": False,
            "is_ai_metadata": False,
            "related_tickets": [],
            "duplicate_prompt_examples": False,
            "date": "2026-05-12",
        }
        reviewer.ai_handler = MagicMock()
        reviewer.ai_handler.chat_completion = AsyncMock(
            return_value=("review:\n  key_issues_to_review: []\n  security_concerns: 'No'\n  relevant_tests: 'No'\n  estimated_effort_to_review_[1-5]: 1", "stop")
        )
        reviewer.incremental = MagicMock()
        reviewer.incremental.is_incremental = False
        from pr_agent.algo.token_handler import TokenHandler
        reviewer.token_handler = MagicMock(spec=TokenHandler)
        reviewer.token_handler.prompt_tokens = 100
        reviewer.pr_description = "desc"
        reviewer.pr_description_files = None
        return reviewer


class TestPRReviewerDelegation:
    def test_agent_mode_false_does_not_delegate(self):
        from pr_agent.tools.pr_reviewer import PRReviewer
        from pr_agent.config_loader import get_settings
        get_settings().pr_reviewer_agent.agent_mode = False
        try:
            reviewer = _make_pr_reviewer(agent_mode=False)
            with patch("pr_agent.tools.pr_reviewer.get_pr_diff", return_value="+ some diff"):
                with patch("pr_agent.tools.pr_reviewer.extract_and_cache_pr_tickets", new=AsyncMock()):
                    with patch("pr_agent.tools.pr_reviewer.retry_with_fallback_models", new=AsyncMock()):
                        with patch("pr_agent.tools.pr_reviewer.AgenticPRReviewer", create=True) as mock_agent:
                            asyncio.run(reviewer.run())
                            mock_agent.assert_not_called()
        finally:
            get_settings().pr_reviewer_agent.agent_mode = False

    def test_agent_mode_true_delegates_to_agentic_reviewer(self):
        from pr_agent.config_loader import get_settings
        get_settings().pr_reviewer_agent.agent_mode = True
        try:
            reviewer = _make_pr_reviewer(agent_mode=True)
            mock_agentic = MagicMock()
            mock_agentic.run = AsyncMock(return_value=None)
            with patch("pr_agent.tools.pr_agentic_reviewer.AgenticPRReviewer", return_value=mock_agentic) as mock_class:
                asyncio.run(reviewer.run())
                mock_class.assert_called_once_with(reviewer)
                mock_agentic.run.assert_called_once()
        finally:
            get_settings().pr_reviewer_agent.agent_mode = False

    def test_delegation_import_is_not_top_level(self):
        """AgenticPRReviewer must not be imported at module level in pr_reviewer.py."""
        import ast
        import pathlib
        source = pathlib.Path("pr_agent/tools/pr_reviewer.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if node.col_offset == 0:
                    names = [alias.name for alias in getattr(node, "names", [])]
                    module = getattr(node, "module", "") or ""
                    assert "pr_agentic_reviewer" not in module and not any(
                        "AgenticPRReviewer" in n for n in names
                    ), "AgenticPRReviewer must not be imported at module level"
