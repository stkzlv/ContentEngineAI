# Update Documentation Playbook

Review and update project documentation for accuracy and quality.

## Process
1. Review branch changes: `git diff main...HEAD`.
2. Analyze documentation structure.
3. Verify statements against actual codebase (using `grep`, `ls`, reading code).
4. Update outdated sections.
5. Remove duplicated information.

## Discover Documentation
```bash
ls *.md              # Root documentation files (GEMINI.md, README.md)
ls docs/*.md         # Extended documentation
```

## Guidelines
- `GEMINI.md`: Keep up-to-date as the primary agent memory.
- `README.md`: Keep short, link to detailed docs.
- Core docs: Focus on permanent features.
- Use collapsible sections for large content: `<details><summary>Title</summary>...</details>`.
- Verify code examples still work.
- Use relative links for internal references.
- Follow GitHub-Flavored Markdown (GFM).

## Review Checklist
- [ ] Statements match current code behavior.
- [ ] Code examples are runnable.
- [ ] No duplicated sections across files.
- [ ] Internal links are valid.
- [ ] Headings follow hierarchy.
- [ ] `CHANGELOG.md` updated for recent changes.
