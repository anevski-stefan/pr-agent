import pytest
from unittest.mock import MagicMock, patch

from pr_agent.tools.ticket_pr_compliance_check import find_jira_tickets, extract_tickets


class TestFindJiraTickets:
    def test_standard_format(self):
        result = find_jira_tickets("Fix PROJ-123 and also TEAM-456")
        assert "PROJ-123" in result
        assert "TEAM-456" in result

    def test_jira_url_format(self):
        result = find_jira_tickets("See https://company.atlassian.net/browse/PROJ-123 for details")
        assert "PROJ-123" in result

    def test_branch_name_prefix(self):
        result = find_jira_tickets("PROJ-123-my-feature-branch")
        assert "PROJ-123" in result

    def test_no_tickets(self):
        result = find_jira_tickets("no tickets here")
        assert result == []

    def test_empty_string(self):
        result = find_jira_tickets("")
        assert result == []

    def test_deduplicates(self):
        result = find_jira_tickets("PROJ-123 and PROJ-123 again")
        assert result.count("PROJ-123") == 1


class TestFetchJiraTicketDetails:
    def test_no_config_returns_none(self, monkeypatch):
        import pr_agent.tools.ticket_pr_compliance_check as m
        fake_settings = MagicMock()
        fake_settings.get = lambda key, default=None: None
        monkeypatch.setattr(m, "get_settings", lambda: fake_settings)

        result = m.fetch_jira_ticket_details("PROJ-123")
        assert result is None

    def test_cloud_email_auth_returns_ticket(self, monkeypatch):
        import pr_agent.tools.ticket_pr_compliance_check as m

        settings_map = {
            "jira.jira_base_url": "https://company.atlassian.net",
            "jira.jira_api_token": "my_token",
            "jira.jira_api_email": "user@company.com",
            "jira.jira_servers": None,
        }
        fake_settings = MagicMock()
        fake_settings.get = lambda key, default=None: settings_map.get(key, default)
        monkeypatch.setattr(m, "get_settings", lambda: fake_settings)

        mock_jira_instance = MagicMock()
        mock_jira_instance.get_issue.return_value = {
            "key": "PROJ-123",
            "fields": {
                "summary": "My ticket title",
                "description": "Ticket body text",
                "labels": ["frontend", "bug"],
            },
        }

        with patch("pr_agent.tools.ticket_pr_compliance_check.AtlassianJira", return_value=mock_jira_instance):
            result = m.fetch_jira_ticket_details("PROJ-123")

        assert result is not None
        assert result["ticket_id"] == "PROJ-123"
        assert result["title"] == "My ticket title"
        assert result["body"] == "Ticket body text"
        assert "frontend" in result["labels"]
        assert result["ticket_url"] == "https://company.atlassian.net/browse/PROJ-123"

    def test_pat_auth_no_email(self, monkeypatch):
        import pr_agent.tools.ticket_pr_compliance_check as m

        settings_map = {
            "jira.jira_base_url": "https://jira.company.com",
            "jira.jira_api_token": "my_pat_token",
            "jira.jira_api_email": None,
            "jira.jira_servers": None,
        }
        fake_settings = MagicMock()
        fake_settings.get = lambda key, default=None: settings_map.get(key, default)
        monkeypatch.setattr(m, "get_settings", lambda: fake_settings)

        mock_jira_instance = MagicMock()
        mock_jira_instance.get_issue.return_value = {
            "key": "MYPROJ-42",
            "fields": {
                "summary": "PAT auth ticket",
                "description": "PAT body",
                "labels": [],
            },
        }

        with patch("pr_agent.tools.ticket_pr_compliance_check.AtlassianJira", return_value=mock_jira_instance):
            result = m.fetch_jira_ticket_details("MYPROJ-42")

        assert result is not None
        assert result["ticket_id"] == "MYPROJ-42"
        assert result["title"] == "PAT auth ticket"

    def test_multi_server_fallback_to_second(self, monkeypatch):
        import pr_agent.tools.ticket_pr_compliance_check as m

        settings_map = {
            "jira.jira_servers": ["https://server1.jira.com", "https://server2.jira.com"],
            "jira.jira_api_token": ["token1", "token2"],
            "jira.jira_api_email": ["user1@example.com", "user2@example.com"],
            "jira.jira_base_url": None,
        }
        fake_settings = MagicMock()
        fake_settings.get = lambda key, default=None: settings_map.get(key, default)
        monkeypatch.setattr(m, "get_settings", lambda: fake_settings)

        mock_fail = MagicMock()
        mock_fail.get_issue.side_effect = Exception("Connection refused")
        mock_ok = MagicMock()
        mock_ok.get_issue.return_value = {
            "key": "PROJ-99",
            "fields": {"summary": "Found on server2", "description": "body", "labels": []},
        }

        call_count = [0]

        def jira_factory(*args, **kwargs):
            call_count[0] += 1
            return mock_fail if call_count[0] == 1 else mock_ok

        with patch("pr_agent.tools.ticket_pr_compliance_check.AtlassianJira", side_effect=jira_factory):
            result = m.fetch_jira_ticket_details("PROJ-99")

        assert result is not None
        assert result["title"] == "Found on server2"

    def test_body_truncated_if_too_long(self, monkeypatch):
        import pr_agent.tools.ticket_pr_compliance_check as m

        settings_map = {
            "jira.jira_base_url": "https://company.atlassian.net",
            "jira.jira_api_token": "token",
            "jira.jira_api_email": "user@company.com",
            "jira.jira_servers": None,
        }
        fake_settings = MagicMock()
        fake_settings.get = lambda key, default=None: settings_map.get(key, default)
        monkeypatch.setattr(m, "get_settings", lambda: fake_settings)

        long_body = "x" * 15000
        mock_jira_instance = MagicMock()
        mock_jira_instance.get_issue.return_value = {
            "key": "PROJ-1",
            "fields": {"summary": "Title", "description": long_body, "labels": []},
        }

        with patch("pr_agent.tools.ticket_pr_compliance_check.AtlassianJira", return_value=mock_jira_instance):
            result = m.fetch_jira_ticket_details("PROJ-1")

        assert result is not None
        assert len(result["body"]) <= 10003  # 10000 + "..."
        assert result["body"].endswith("...")


class TestExtractTicketsBitbucket:
    @pytest.mark.asyncio
    async def test_bitbucket_cloud_returns_jira_tickets(self, monkeypatch):
        from pr_agent.git_providers.bitbucket_provider import BitbucketProvider
        import pr_agent.tools.ticket_pr_compliance_check as m

        mock_provider = MagicMock(spec=BitbucketProvider)
        mock_provider.get_user_description.return_value = "Fixes PROJ-123 as discussed"
        mock_provider.get_pr_branch.return_value = "feature/other-branch"

        mock_ticket = {
            "ticket_id": "PROJ-123",
            "ticket_url": "https://company.atlassian.net/browse/PROJ-123",
            "title": "My ticket",
            "body": "Ticket body",
            "labels": "bug",
            "requirements": "",
        }
        monkeypatch.setattr(m, "fetch_jira_ticket_details", lambda ticket_id: mock_ticket)

        result = await extract_tickets(mock_provider)

        assert result is not None
        assert len(result) == 1
        assert result[0]["ticket_id"] == "PROJ-123"

    @pytest.mark.asyncio
    async def test_bitbucket_server_returns_jira_tickets(self, monkeypatch):
        from pr_agent.git_providers.bitbucket_server_provider import BitbucketServerProvider
        import pr_agent.tools.ticket_pr_compliance_check as m

        mock_provider = MagicMock(spec=BitbucketServerProvider)
        mock_provider.get_user_description.return_value = "See TEAM-456 for requirements"
        mock_provider.get_pr_branch.return_value = "main"

        mock_ticket = {
            "ticket_id": "TEAM-456",
            "ticket_url": "https://jira.company.com/browse/TEAM-456",
            "title": "Team ticket",
            "body": "Body",
            "labels": "",
            "requirements": "",
        }
        monkeypatch.setattr(m, "fetch_jira_ticket_details", lambda ticket_id: mock_ticket)

        result = await extract_tickets(mock_provider)

        assert result is not None
        assert len(result) == 1
        assert result[0]["ticket_id"] == "TEAM-456"

    @pytest.mark.asyncio
    async def test_bitbucket_deduplicates_description_and_branch(self, monkeypatch):
        from pr_agent.git_providers.bitbucket_provider import BitbucketProvider
        import pr_agent.tools.ticket_pr_compliance_check as m

        mock_provider = MagicMock(spec=BitbucketProvider)
        mock_provider.get_user_description.return_value = "Fixes PROJ-123"
        mock_provider.get_pr_branch.return_value = "PROJ-123-my-feature"

        fetch_calls = []

        def mock_fetch(ticket_id):
            fetch_calls.append(ticket_id)
            return {"ticket_id": ticket_id, "ticket_url": "", "title": "T", "body": "", "labels": "", "requirements": ""}

        monkeypatch.setattr(m, "fetch_jira_ticket_details", mock_fetch)

        result = await extract_tickets(mock_provider)

        assert result is not None
        assert len(result) == 1
        assert fetch_calls.count("PROJ-123") == 1

    @pytest.mark.asyncio
    async def test_bitbucket_limits_to_3_tickets(self, monkeypatch):
        from pr_agent.git_providers.bitbucket_provider import BitbucketProvider
        import pr_agent.tools.ticket_pr_compliance_check as m

        mock_provider = MagicMock(spec=BitbucketProvider)
        mock_provider.get_user_description.return_value = "PROJ-1 PROJ-2 PROJ-3 PROJ-4"
        mock_provider.get_pr_branch.return_value = "main"

        def mock_fetch(ticket_id):
            return {"ticket_id": ticket_id, "ticket_url": "", "title": f"T{ticket_id}", "body": "", "labels": "", "requirements": ""}

        monkeypatch.setattr(m, "fetch_jira_ticket_details", mock_fetch)

        result = await extract_tickets(mock_provider)

        assert result is not None
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_bitbucket_no_tickets_returns_empty(self, monkeypatch):
        from pr_agent.git_providers.bitbucket_provider import BitbucketProvider
        import pr_agent.tools.ticket_pr_compliance_check as m

        mock_provider = MagicMock(spec=BitbucketProvider)
        mock_provider.get_user_description.return_value = "No ticket references here"
        mock_provider.get_pr_branch.return_value = "feature/my-feature"

        result = await extract_tickets(mock_provider)

        assert not result

    @pytest.mark.asyncio
    async def test_bitbucket_skips_failed_jira_fetch(self, monkeypatch):
        from pr_agent.git_providers.bitbucket_provider import BitbucketProvider
        import pr_agent.tools.ticket_pr_compliance_check as m

        mock_provider = MagicMock(spec=BitbucketProvider)
        mock_provider.get_user_description.return_value = "PROJ-1 PROJ-2"
        mock_provider.get_pr_branch.return_value = "main"

        def mock_fetch(ticket_id):
            if ticket_id == "PROJ-1":
                return None
            return {"ticket_id": ticket_id, "ticket_url": "", "title": "T2", "body": "", "labels": "", "requirements": ""}

        monkeypatch.setattr(m, "fetch_jira_ticket_details", mock_fetch)

        result = await extract_tickets(mock_provider)

        assert result is not None
        assert len(result) == 1
        assert result[0]["ticket_id"] == "PROJ-2"
