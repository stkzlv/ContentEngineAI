Review and update project documentation for accuracy and quality.

## Process
1. Review branch changes: `git diff main...HEAD`
2. Analyze documentation structure
3. Verify statements against actual codebase
4. Update outdated sections
5. Remove duplicated information

## Discover Documentation
```bash
ls *.md              # Root documentation files
ls docs/*.md         # Extended documentation
```

## Guidelines
- README.md: Keep short, link to detailed docs
- Core docs: Focus on permanent features
- Use collapsible sections for large content: `<details><summary>Title</summary>...</details>`
- Verify code examples still work
- Use relative links for internal references
- Follow GitHub-Flavored Markdown (GFM)

## Review Checklist
- [ ] Statements match current code behavior
- [ ] Code examples are runnable
- [ ] No duplicated sections across files
- [ ] Internal links are valid
- [ ] Headings follow hierarchy (no skipped levels)
- [ ] CHANGELOG updated for recent changes
