from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pr_agent.algo.skill_loader import SkillDefinition, get_skill_descriptions_for_triage, load_review_skills

FIXTURES = Path(__file__).parent.parent / "fixtures" / "skills"

SECURITY_MD = (FIXTURES / "security_skill.md").read_text()
FRONTEND_MD = (FIXTURES / "frontend_skill.md").read_text()


def _mock_provider(responses: dict) -> MagicMock:
    """Build a git provider mock where responses maps file_path -> content ('' = not found)."""
    provider = MagicMock()
    provider.get_pr_file_content.side_effect = lambda path, branch: responses.get(path, "")
    return provider


class TestLoadReviewSkills:
    def test_empty_enabled_names_returns_empty(self):
        provider = _mock_provider({})
        result = load_review_skills([], provider, "main")
        assert result == []
        provider.get_pr_file_content.assert_not_called()

    def test_fetches_correct_path_for_each_skill(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
        })
        load_review_skills(["security-and-hardening"], provider, "main")
        provider.get_pr_file_content.assert_called_once_with(
            ".claude/skills/security-and-hardening/SKILL.md", "main"
        )

    def test_parses_name_and_description_from_frontmatter(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
        })
        skills = load_review_skills(["security-and-hardening"], provider, "main")
        assert len(skills) == 1
        assert skills[0].name == "security-and-hardening"
        assert "vulnerabilities" in skills[0].description

    def test_content_populated(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
        })
        skills = load_review_skills(["security-and-hardening"], provider, "main")
        assert skills[0].content  
        assert "Security" in skills[0].content

    def test_skill_not_found_returns_empty_string_skipped(self):
        provider = _mock_provider({})
        skills = load_review_skills(["nonexistent-skill"], provider, "main")
        assert skills == []

    def test_multiple_skills_loaded(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
            ".claude/skills/frontend-ui-engineering/SKILL.md": FRONTEND_MD,
        })
        skills = load_review_skills(
            ["security-and-hardening", "frontend-ui-engineering"], provider, "main"
        )
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"security-and-hardening", "frontend-ui-engineering"}

    def test_one_missing_skill_does_not_affect_others(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
        })
        skills = load_review_skills(
            ["security-and-hardening", "nonexistent-skill"], provider, "main"
        )
        assert len(skills) == 1
        assert skills[0].name == "security-and-hardening"

    def test_skill_without_frontmatter_is_skipped(self):
        no_frontmatter = "# Just a heading\n\nSome content without frontmatter."
        provider = _mock_provider({
            ".claude/skills/bare-skill/SKILL.md": no_frontmatter,
        })
        skills = load_review_skills(["bare-skill"], provider, "main")
        assert skills == []

    def test_content_truncated_to_5k_tokens(self):
        long_body = "word " * 6000
        long_md = f"---\nname: big-skill\ndescription: A very long skill.\n---\n\n{long_body}"
        provider = _mock_provider({
            ".claude/skills/big-skill/SKILL.md": long_md,
        })
        skills = load_review_skills(["big-skill"], provider, "main")
        assert len(skills) == 1
        assert len(skills[0].content) < len(long_md)

    def test_duplicate_names_loaded_only_once(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
        })
        skills = load_review_skills(
            ["security-and-hardening", "security-and-hardening"], provider, "main"
        )
        assert len(skills) == 1
        provider.get_pr_file_content.assert_called_once()

    def test_path_traversal_name_is_rejected(self):
        provider = _mock_provider({})
        for bad_name in ["../../.secrets", "../etc/passwd", "some/nested/skill"]:
            skills = load_review_skills([bad_name], provider, "main")
            assert skills == [], f"Expected '{bad_name}' to be rejected"
        provider.get_pr_file_content.assert_not_called()

    def test_multiline_description_parsed(self):
        multiline_md = (
            "---\n"
            "name: my-skill\n"
            "description: |\n"
            "  First line of description.\n"
            "  Second line of description.\n"
            "---\n\n"
            "# Body"
        )
        provider = _mock_provider({".claude/skills/my-skill/SKILL.md": multiline_md})
        skills = load_review_skills(["my-skill"], provider, "main")
        assert len(skills) == 1
        assert "First line" in skills[0].description

    def test_branch_passed_to_provider(self):
        provider = _mock_provider({
            ".claude/skills/security-and-hardening/SKILL.md": SECURITY_MD,
        })
        load_review_skills(["security-and-hardening"], provider, "feature/my-branch")
        provider.get_pr_file_content.assert_called_once_with(
            ".claude/skills/security-and-hardening/SKILL.md", "feature/my-branch"
        )


class TestGetSkillDescriptionsForTriage:
    def test_empty_list_returns_empty_string(self):
        assert get_skill_descriptions_for_triage([]) == ""

    def test_single_skill_formatted_correctly(self):
        skill = SkillDefinition(
            name="security-and-hardening",
            description="Hardens code against vulnerabilities.",
            content="full content here",
        )
        result = get_skill_descriptions_for_triage([skill])
        assert result == "- security-and-hardening: Hardens code against vulnerabilities."

    def test_multiple_skills_one_per_line(self):
        skills = [
            SkillDefinition(name="security-and-hardening", description="Security checks.", content=""),
            SkillDefinition(name="frontend-ui-engineering", description="UI quality.", content=""),
        ]
        result = get_skill_descriptions_for_triage(skills)
        lines = result.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("- security-and-hardening:")
        assert lines[1].startswith("- frontend-ui-engineering:")
