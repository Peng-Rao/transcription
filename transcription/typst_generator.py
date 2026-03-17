import logging
import os
import re
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)


class TypstGenerator:
    def __init__(self, api_key=None, model="deepseek-reasoner"):
        self.model = model
        self.client = None

        if api_key:
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_notes(self, processed_text, output_path, title="Lecture Notes"):
        """Generate Typst lecture notes from processed text."""
        logger.info("Generating Typst notes...")
        prompt = self._create_prompt(processed_text, title)

        try:
            if self.client:
                typst_content = self._generate_with_llm(prompt)
            else:
                logger.warning("No API key provided, generating template Typst notes")
                typst_content = self._generate_template_notes(processed_text, title)

            typst_content = self._sanitize_typst_content(typst_content)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(typst_content)

            logger.info(f"Typst notes generated: {output_path}")
        except Exception as e:
            logger.error(f"Error generating Typst notes: {e}")
            typst_content = self._generate_template_notes(processed_text, title)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(typst_content)

    def _create_prompt(
        self, text, title, template_filename="prompt_template_typst.txt"
    ):
        """Create prompt from template without using str.format on Typst braces."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, template_filename)

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        date_str = datetime.now().strftime("%B %d, %Y")

        # Avoid Python format parsing because Typst reference snippets often
        # contain braces that are not formatting placeholders.
        prompt = template
        prompt = prompt.replace("{title}", title)
        prompt = prompt.replace("{date}", date_str)
        prompt = prompt.replace("{transcript}", text)

        # If the updated template does not define placeholders, append metadata
        # and transcript so the model still receives source content.
        if "{transcript}" not in template:
            prompt = (
                f"{prompt}\n\n"
                f"Document metadata:\n"
                f"- Title: {title}\n"
                f"- Date: {date_str}\n\n"
                f"Transcript:\n{text}\n"
            )

        return prompt

    def _generate_with_llm(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert lecture transcripts into polished Typst notes. "
                        "Return only valid Typst code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        content = response.choices[0].message.content or ""

        # Normalize common fenced-code outputs to raw Typst content.
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()

        return content

    def _sanitize_typst_content(self, content):
        """Patch frequent LLM Typst syntax issues in ket/bra notations."""
        if not content:
            return content

        fixed = content

        # Typst function calls cannot pass comma-separated signs as bare tokens.
        # Convert ket(+, -) -> ket("+, -") and same for bra().
        fixed = re.sub(
            r"\bket\(\s*([+-])\s*,\s*([+-])\s*\)",
            lambda m: f'ket("{m.group(1)}, {m.group(2)}")',
            fixed,
        )
        fixed = re.sub(
            r"\bbra\(\s*([+-])\s*,\s*([+-])\s*\)",
            lambda m: f'bra("{m.group(1)}, {m.group(2)}")',
            fixed,
        )

        # Normalize common two-sign state labels used as bare expressions.
        fixed = re.sub(r"\bket\(\s*\+\+\s*\)", 'ket("++")', fixed)
        fixed = re.sub(r"\bket\(\s*\+-\s*\)", 'ket("+-")', fixed)
        fixed = re.sub(r"\bket\(\s*-\+\s*\)", 'ket("-+")', fixed)
        fixed = re.sub(r"\bket\(\s*--\s*\)", 'ket("--")', fixed)

        # Remove common LaTeX size wrappers that are invalid in Typst math.
        fixed = re.sub(r"\bbigl\(", "(", fixed)
        fixed = re.sub(r"\bbigr\)", ")", fixed)

        return fixed

    def _generate_template_notes(
        self, text, title, template_filename="template_typst.typ"
    ):
        """Generate a minimal Typst document without LLM."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, template_filename)

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = []
        for i, para in enumerate(paragraphs[:6], 1):
            sections.append(f"== Topic {i}\n\n{para}")

        body = "\n\n".join(sections) if sections else "No content available."

        return template.format(
            title=title,
            date=datetime.now().strftime("%B %d, %Y"),
            body=body,
        )
