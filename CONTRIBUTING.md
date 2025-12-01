# Contributing Guidelines

## Branch Strategy

- `main` - Stable, production-ready code
- `develop` - Integration branch for features
- `feature/*` - Individual feature branches
- `fix/*` - Bug fix branches

## Workflow

1. Create a feature branch from `develop`
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit
   ```bash
   git add .
   git commit -m "feat: description of your feature"
   ```

3. Push and create a Pull Request
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `style:` - Code style changes (formatting)
- `chore:` - Maintenance tasks

## Code Style

- Use Black for Python formatting
- Follow PEP 8 guidelines
- Add docstrings to all functions and classes
- Include type hints

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Run: `pytest tests/ -v`

## Review Process

- All PRs require at least one approval
- Address review comments before merging
- Squash commits when merging to `develop`
