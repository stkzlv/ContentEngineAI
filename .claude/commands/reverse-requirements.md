Generate requirements documentation from existing implementation.

## Purpose
Reverse-engineer high-level requirements by analyzing actual code behavior. Useful when implementation exists but requirements are missing or outdated.

## Process
1. Identify the feature/module to document
2. Explore the implementation thoroughly (config, code, tests)
3. Extract key behaviors, defaults, and options
4. Write human-readable requirements (not implementation details)
5. Update `docs/requirements.md` with new sections

## What to Capture
- **Behaviors**: What the system does, not how
- **Defaults**: Default values users should know about
- **Options**: Configurable parameters and their allowed values
- **Constraints**: Limits, validation rules, edge cases
- **Dependencies**: What features depend on each other

## Writing Style
- Keep requirements short and scannable
- Use bullet points, not paragraphs
- Include defaults in parentheses: "Width as percentage (default 75%)"
- Bold key terms: "**Aspect modes**: letterbox, crop-to-fit"
- Avoid code snippets - link to docs instead

## Example
From implementation:
```python
image_width_percent: float = Field(0.75)
image_top_position_percent: float = Field(0.20)
preserve_aspect_ratio: bool = Field(True)
```

To requirement:
```markdown
### Image Positioning
- Width as percentage of frame (default 75%)
- Vertical position from top (default 20%)
- Preserve aspect ratio by default
```

## Target File
Update: `docs/requirements.md`
