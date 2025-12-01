# Configuration Files

This directory contains configuration files for different components of the system.

## Structure

- `agents/` - Agent-specific configurations
- `evaluation/` - Evaluation metric configurations
- `mcp/` - MCP server configurations
- `models/` - LLM model configurations

All configs use YAML or JSON format.

## Example

```yaml
# configs/agents/question_generator.yaml
model:
  name: gpt-4-turbo
  temperature: 0.7
  max_tokens: 2000

diversity:
  content_threshold: 0.85
  type_distribution:
    factoid: 0.15
    conceptual: 0.20
    reasoning: 0.25
```
