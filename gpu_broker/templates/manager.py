"""Template manager for GPU Broker JSONata task templates."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

JSONATA_MODULE = "/home/lyweiwei/.nvm/versions/node/v24.14.1/lib/node_modules/jsonata"

NODE_SCRIPT = """
const jsonata = require('JSONATA_PATH');
let input = '';
process.stdin.on('data', d => input += d);
process.stdin.on('end', async () => {
    const {template, variables} = JSON.parse(input);
    try {
        const expr = jsonata(template);
        const result = await expr.evaluate(variables);
        process.stdout.write(JSON.stringify(result));
    } catch(e) {
        process.stderr.write(JSON.stringify({error: e.message}));
        process.exit(1);
    }
});
""".replace('JSONATA_PATH', JSONATA_MODULE)


def evaluate_jsonata(template_expr: str, variables: dict) -> dict:
    """Evaluate a JSONata expression with given variables using Node.js subprocess."""
    proc = subprocess.run(
        ["node", "-e", NODE_SCRIPT],
        input=json.dumps({"template": template_expr, "variables": variables}),
        capture_output=True, text=True, timeout=10
    )
    if proc.returncode != 0:
        error = proc.stderr
        try:
            error = json.loads(error).get("error", error)
        except Exception:
            pass
        raise RuntimeError(f"JSONata evaluation failed: {error}")
    return json.loads(proc.stdout)


class TemplateManager:
    """Manages task templates stored as YAML files."""

    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.builtin_dir = Path(__file__).parent / 'builtin'
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_templates()

    def list(self, tag: str = None, search: str = None) -> list:
        """List templates, optionally filtered by tag and/or keyword search.

        Returns a list of metadata dicts (without the template body).
        """
        results = []
        for yaml_file in sorted(self.templates_dir.glob("*.yaml")):
            try:
                data = self._load_template(yaml_file)
            except Exception:
                continue

            # Tag filter
            if tag:
                tags = data.get("tags", []) or []
                if tag not in tags:
                    continue

            # Keyword search in name and description
            if search:
                search_lower = search.lower()
                name = (data.get("name") or "").lower()
                description = (data.get("description") or "").lower()
                tags_str = " ".join(data.get("tags") or []).lower()
                if search_lower not in name and search_lower not in description and search_lower not in tags_str:
                    continue

            # Return metadata without template body
            meta = {
                "name": data.get("name"),
                "description": data.get("description"),
                "tags": data.get("tags", []),
                "author": data.get("author"),
                "variables": data.get("variables", {}),
                "file": yaml_file.name,
            }
            results.append(meta)
        return results

    def get(self, name: str) -> dict | None:
        """Get full template details including template body."""
        path = self._find_template_file(name)
        if path is None:
            return None
        data = self._load_template(path)
        data["file"] = path.name
        return data

    def create(self, name: str, content: str) -> dict:
        """Create or update a template from YAML content string."""
        # Parse to validate YAML
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict):
            raise ValueError("Template content must be a YAML mapping")

        # Use slugified name as filename
        slug = self._slugify(name)
        file_path = self.templates_dir / f"{slug}.yaml"
        file_path.write_text(content, encoding="utf-8")

        # Return the parsed template with file info
        result = self._load_template(file_path)
        result["file"] = file_path.name
        return result

    def delete(self, name: str) -> bool:
        """Delete a template by name. Returns True if deleted, False if not found."""
        path = self._find_template_file(name)
        if path is None:
            return False
        path.unlink()
        return True

    def render(self, name: str, variables: dict) -> dict:
        """Render a template: load JSONata template, inject variables, return task JSON."""
        data = self.get(name)
        if data is None:
            raise ValueError(f"Template '{name}' not found")

        # Validate required variables
        is_valid, missing = self.validate(name, variables)
        if not is_valid:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")

        # Merge defaults into variables
        merged = {}
        var_defs = data.get("variables") or {}
        for var_name, var_info in var_defs.items():
            if isinstance(var_info, dict):
                default = var_info.get("default")
            else:
                default = None
            merged[var_name] = default

        # User-provided variables override defaults
        merged.update(variables)

        # Remove None values to avoid JSONata issues with null bindings
        # but keep explicit nulls from defaults if user didn't override
        template_expr = data.get("template", "")
        result = evaluate_jsonata(template_expr, merged)
        return result

    def validate(self, name: str, variables: dict) -> tuple:
        """Validate that required variables are provided.

        Returns (is_valid: bool, missing_fields: list[str]).
        """
        data = self.get(name)
        if data is None:
            raise ValueError(f"Template '{name}' not found")

        var_defs = data.get("variables") or {}
        missing = []
        for var_name, var_info in var_defs.items():
            if isinstance(var_info, dict) and var_info.get("required"):
                if var_name not in variables or variables[var_name] is None:
                    missing.append(var_name)

        return (len(missing) == 0, missing)

    def _ensure_templates(self):
        """Copy builtin templates to user templates dir if it's empty."""
        existing = list(self.templates_dir.glob("*.yaml"))
        if existing:
            return
        if self.builtin_dir.exists():
            for builtin_file in self.builtin_dir.glob("*.yaml"):
                dest = self.templates_dir / builtin_file.name
                shutil.copy2(builtin_file, dest)

    def _slugify(self, name: str) -> str:
        """Convert name to a slug: lowercase, spaces to hyphens, only [a-z0-9-]."""
        slug = name.lower().replace(" ", "-")
        slug = re.sub(r"[^a-z0-9-]", "", slug)
        return slug

    def _find_template_file(self, name: str) -> Path | None:
        """Find a template file by name. Tries exact filename match first, then slugify."""
        # Try exact match as given (with .yaml)
        exact = self.templates_dir / f"{name}.yaml"
        if exact.exists():
            return exact

        # Try slugified version
        slug = self._slugify(name)
        slug_path = self.templates_dir / f"{slug}.yaml"
        if slug_path.exists():
            return slug_path

        # Try matching by name field inside yaml files
        for yaml_file in self.templates_dir.glob("*.yaml"):
            try:
                data = self._load_template(yaml_file)
                if data.get("name") == name:
                    return yaml_file
            except Exception:
                continue

        return None

    def _load_template(self, path: Path) -> dict:
        """Load and parse a YAML template file."""
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid template file: {path}")
        return data
