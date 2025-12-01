# Quick Start: Push to GitHub

## Step 1: Verify Git Setup

```bash
cd c:\Users\benam\Downloads\Agentic_AI
git status
```

You should see: "On branch main" or "On branch master"

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "feat: initial project structure with multi-track organization

- Created folder structure for 4 research tracks
- generation/ for data generation methods (multi-agent, etc.)
- evaluation/ for metrics research
- taxonomy/ for question classification
- multimodal/ for visual document processing
- shared/ for common utilities
- docs/ with complete architecture and collaboration guides
- Added comprehensive READMEs for each track
- Set up CI/CD workflows"
```

## Step 4: Push to GitHub

```bash
# Verify remote
git remote -v

# Should show:
# origin  https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems.git (fetch)
# origin  https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems.git (push)

# Push to main branch
git branch -M main
git push -u origin main
```

## Step 5: Create Development Branch

```bash
# Create develop branch
git checkout -b develop
git push -u origin develop

# Go back to main
git checkout main
```

## Step 6: Set Up Branch Protection (on GitHub)

1. Go to: https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems/settings/branches
2. Click "Add rule"
3. Branch name pattern: `main`
4. Enable:
   - ☑ Require a pull request before merging
   - ☑ Require approvals (at least 1)
5. Save changes

Now team members **must** use pull requests to merge to main!

## Step 7: Invite Collaborators

1. Go to: https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems/settings/access
2. Click "Add people"
3. Enter GitHub usernames
4. Select role: "Write" (can push branches and create PRs)

## Step 8: Share With Team

Send them:
1. **Repository URL:** https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems
2. **Collaboration Guide:** `docs/COLLABORATION_GUIDE.md`
3. **Their assigned folder:**
   - Evaluation researcher → `evaluation/`
   - Taxonomy researcher → `taxonomy/`
   - Multimodal researcher → `multimodal/`

## Next Steps for You (Aziz)

```bash
# Start working on your first feature
git checkout -b generation/setup-mcp-server

# Work on your code in generation/multi-agent/

# When done, commit and push
git add generation/multi-agent/
git commit -m "feat(generation): set up MCP server for textbook processing"
git push origin generation/setup-mcp-server

# Then create PR on GitHub
```

## Verification Checklist

- [ ] Repository shows all files on GitHub
- [ ] README displays correctly
- [ ] Collaborators invited
- [ ] Branch protection enabled on `main`
- [ ] `develop` branch created
- [ ] Team members can clone the repo
- [ ] `.env.example` exists (no actual `.env` committed!)

## Troubleshooting

### "Permission denied" when pushing
→ Check GitHub authentication. Use Personal Access Token or SSH key.

### "Repository not found"
→ Verify you're the owner or have been added as collaborator.

### "Large files rejected"
→ Check `.gitignore` is working. Don't commit PDFs, models, or datasets.

```bash
# See what's being tracked
git ls-files

# Remove large file from git (if committed by mistake)
git rm --cached path/to/large/file
git commit -m "remove large file"
```

### Want to start fresh?
```bash
# DANGER: This deletes everything and starts over
rm -rf .git
git init
git remote add origin https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems.git
# Then repeat Step 2-4
```
