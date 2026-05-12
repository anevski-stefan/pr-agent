from dataclasses import dataclass

from pr_agent.algo.utils import clip_tokens
from pr_agent.log import get_logger

SKILL_MAX_TOKENS = 5000


@dataclass
class SkillDefinition:
    name: str
    description: str
    content: str


def _parse_frontmatter(raw: str) -> tuple[str, str]:
    """Extract name and description from YAML frontmatter delimited by ---.

    Manual line-by-line parse to avoid adding a yaml dependency.
    Handles single-line and multi-line (block scalar) description values by
    collecting continuation lines until the next key or end of frontmatter.
    """
    if not raw.startswith("---"):
        return "", ""

    end = raw.find("\n---", 3)
    if end == -1:
        return "", ""

    frontmatter = raw[3:end]
    name = ""
    description_lines: list[str] = []
    in_description = False

    for line in frontmatter.splitlines():
        if line.startswith("name:"):
            name = line[len("name:"):].strip()
            in_description = False
        elif line.startswith("description:"):
            value = line[len("description:"):].strip()
            description_lines = [value] if value else []
            in_description = True
        elif in_description and line[:1].isspace():
            description_lines.append(line.strip())
        else:
            in_description = False

    description = " ".join(description_lines).strip()
    return name, description


def load_review_skills(enabled_names: list[str], git_provider, branch: str) -> list[SkillDefinition]:
    """Fetch and parse SKILL.md files from .claude/skills/ via the git provider API."""
    unique_names = list(dict.fromkeys(enabled_names))
    if not unique_names:
        return []

    skills = []
    for name in unique_names:
        if ".." in name or "/" in name:
            get_logger().warning(f"Skill name '{name}' contains invalid characters, skipping")
            continue

        path = f".claude/skills/{name}/SKILL.md"
        raw = git_provider.get_pr_file_content(path, branch)

        if not raw:
            get_logger().warning(f"Skill '{name}' not found at {path}, skipping")
            continue

        skill_name, description = _parse_frontmatter(raw)
        if not skill_name:
            get_logger().warning(f"Skill '{name}' has no valid frontmatter, skipping")
            continue

        content = clip_tokens(raw, SKILL_MAX_TOKENS)
        skills.append(SkillDefinition(name=skill_name, description=description, content=content))

    return skills


def get_skill_descriptions_for_triage(skills: list[SkillDefinition]) -> str:
    """Format skill name+description pairs for the triage prompt."""
    return "\n".join(f"- {s.name}: {s.description}" for s in skills)
