import logging
import os
import re
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)


class LaTeXGenerator:
    def __init__(self, api_key=None, model="deepseek-reasoner"):
        """
        Initialize LaTeX generator with DeepSeek API

        Args:
            api_key (str): DeepSeek API key
            model (str): Model to use for generation (default: deepseek-reasoner)
        """
        self.model = model
        self.client = None

        if api_key:
            # Create DeepSeek client with proper base URL
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_notes(self, processed_text, output_path, title="Lecture Notes"):
        """Generate LaTeX notes from processed text (same as before)"""
        logger.info("Generating LaTeX notes...")
        prompt = self._create_prompt(processed_text, title)

        try:
            # Use DeepSeek client if available
            if self.client:
                latex_content = self._generate_with_deepseek(prompt)
            else:
                logger.warning("No API key provided, generating template notes")
                latex_content = self._generate_template_notes(processed_text, title)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(latex_content)

            logger.info(f"LaTeX notes generated: {output_path}")

        except Exception as e:
            logger.error(f"Error generating notes: {e}")
            latex_content = self._generate_template_notes(processed_text, title)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(latex_content)

    def _create_prompt(self, text, title, template_filename="prompt_template.txt"):
        """Create prompt by loading template from file and filling in values"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(current_dir, template_filename)

            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            logger.error(f"Failed to read prompt template: {e}")
            raise
        return template

    def _generate_with_deepseek(self, prompt):
        """Generate notes using DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating well-structured LaTeX documents from lecture transcripts. Create professional, academic notes with proper formatting.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

    def _generate_template_notes(
        self, text, title, template_filename="template_latex.tex"
    ):
        """Generate template LaTeX notes from local file"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(current_dir, template_filename)

            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            logger.error(f"Failed to read LaTeX template: {e}")
            raise

        # Process content: take first 5 non-empty paragraphs as sections
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        body = ""
        for i, para in enumerate(paragraphs[:5], 1):
            body += (
                f"\\section{{Topic {i}}}\n{self._format_paragraph_for_latex(para)}\n\n"
            )

        # Fill in the template
        latex_content = template.format(
            title=title,
            date=datetime.now().strftime("%B %d, %Y"),
            body=body,
        )

        return latex_content

    def _format_paragraph_for_latex(self, paragraph):
        """Basic formatting of paragraph for LaTeX"""
        # Escape special LaTeX characters
        paragraph = paragraph.replace("&", "\\&")
        paragraph = paragraph.replace("%", "\\%")
        paragraph = paragraph.replace("$", "\\$")
        paragraph = paragraph.replace("#", "\\#")
        paragraph = paragraph.replace("_", "\\_")
        paragraph = paragraph.replace("{", "\\{")
        paragraph = paragraph.replace("}", "\\}")

        # Add some basic formatting
        # Bold important terms (simple heuristic)
        paragraph = re.sub(
            r"\b(definition|theorem|important|key|main|primary)\b",
            r"\\textbf{\1}",
            paragraph,
            flags=re.IGNORECASE,
        )

        return paragraph
