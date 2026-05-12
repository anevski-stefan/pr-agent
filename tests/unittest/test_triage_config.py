from jinja2 import Environment, StrictUndefined

from pr_agent.config_loader import get_settings


class TestTriageConfiguration:
    def test_agent_mode_defaults_false(self):
        assert get_settings().pr_reviewer_agent.agent_mode is False

    def test_agent_max_skill_calls_defaults_3(self):
        assert get_settings().pr_reviewer_agent.agent_max_skill_calls == 3

    def test_agent_min_risk_to_trigger_defaults_3(self):
        assert get_settings().pr_reviewer_agent.agent_min_risk_to_trigger == 3

    def test_agent_parallel_skills_defaults_true(self):
        assert get_settings().pr_reviewer_agent.agent_parallel_skills is True

    def test_agent_max_findings_per_skill_defaults_5(self):
        assert get_settings().pr_reviewer_agent.agent_max_findings_per_skill == 5

    def test_agent_triage_model_defaults_empty(self):
        assert get_settings().pr_reviewer_agent.agent_triage_model == ""

    def test_agent_skill_call_model_defaults_empty(self):
        assert get_settings().pr_reviewer_agent.agent_skill_call_model == ""

    def test_agent_review_skills_has_7_defaults(self):
        skills = get_settings().pr_reviewer_agent.agent_review_skills
        assert len(skills) == 7

    def test_agent_review_skills_contains_expected_names(self):
        skills = get_settings().pr_reviewer_agent.agent_review_skills
        expected = {
            "code-review-and-quality",
            "security-and-hardening",
            "performance-optimization",
            "frontend-ui-engineering",
            "api-and-interface-design",
            "code-simplification",
            "test-driven-development",
        }
        assert set(skills) == expected


class TestTriagePrompt:
    def test_triage_prompt_system_exists(self):
        system = get_settings().pr_reviewer_triage_prompt.system
        assert system and len(system) > 50

    def test_triage_prompt_user_exists(self):
        user = get_settings().pr_reviewer_triage_prompt.user
        assert user and len(user) > 20

    def test_user_prompt_contains_diff_variable(self):
        assert "{{ diff" in get_settings().pr_reviewer_triage_prompt.user

    def test_user_prompt_contains_skill_descriptions_variable(self):
        assert "{{ skill_descriptions" in get_settings().pr_reviewer_triage_prompt.user

    def test_user_prompt_contains_agent_max_skill_calls_variable(self):
        assert "{{ agent_max_skill_calls" in get_settings().pr_reviewer_triage_prompt.user

    def test_system_prompt_mentions_file_risk_scores(self):
        assert "file_risk_scores" in get_settings().pr_reviewer_triage_prompt.system

    def test_system_prompt_mentions_skill_triggers(self):
        assert "skill_triggers" in get_settings().pr_reviewer_triage_prompt.system

    def test_system_prompt_mentions_initial_findings(self):
        assert "initial_findings" in get_settings().pr_reviewer_triage_prompt.system

    def test_system_prompt_has_no_template_variables(self):
        assert "{{" not in get_settings().pr_reviewer_triage_prompt.system

    def test_triage_prompt_renders_without_error(self):
        env = Environment(undefined=StrictUndefined)
        variables = {
            "diff": "## File: 'src/auth.py'\n+def login(): pass",
            "skill_descriptions": "- security-and-hardening: Hardens code.",
            "agent_max_skill_calls": 3,
            "title": "Add login endpoint",
            "branch": "feature/login",
            "description": "Adds a login route",
            "date": "2026-05-12",
        }
        system = get_settings().pr_reviewer_triage_prompt.system
        user = get_settings().pr_reviewer_triage_prompt.user
        env.from_string(system).render(variables)
        env.from_string(user).render(variables)
