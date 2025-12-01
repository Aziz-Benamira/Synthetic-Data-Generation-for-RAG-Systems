# GitHub Collaboration Guide

**Repository:** Synthetic-Data-Generation-for-RAG-Systems  
**Team:** Multi-track research project  
**Strategy:** Folder-based organization + Feature branches

---

## 🎯 Repository Structure

```
Synthetic-Data-Generation-for-RAG-Systems/
├── generation/           # Data generation methods
│   └── multi-agent/     # Aziz's multi-agent approach
├── evaluation/           # Evaluation metrics research
├── taxonomy/             # Question taxonomy research
├── multimodal/           # Multimodal RAG research
├── shared/               # Shared utilities and data
├── docs/                 # Documentation
└── .github/              # GitHub workflows
```

---

## 📋 Workflow for Team Members

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Aziz-Benamira/Synthetic-Data-Generation-for-RAG-Systems.git
cd Synthetic-Data-Generation-for-RAG-Systems
```

### **Step 2: Choose Your Work Directory**

Each team member works in their designated folder:

| Team Member | Focus Area | Directory |
|------------|------------|-----------|
| **Aziz** | Multi-agent generation | `generation/multi-agent/` |
| **Yassine** | Graph based generation | `generation/graph_based/` |
| **Maloe** | Evaluation metrics | `evaluation/` |
| **Ameni** | Question taxonomy | `taxonomy/` |
| **Seif** | Multimodal RAG | `multimodal/` |

### **Step 3: Create a Feature Branch**

**IMPORTANT:** Always create a branch for your work, never commit directly to `main`.

```bash
# Naming convention: <area>/<feature-name>
# Examples:
git checkout -b generation/reflexion-agent        
git checkout -b evaluation/ragas-metrics         
git checkout -b taxonomy/question-classifier      
git checkout -b multimodal/image-extraction      
```

### **Step 4: Work on Your Feature**

```bash
# Make your changes in your designated folder
cd generation/multi-agent/  # or your folder

# Check status
git status

# Add your changes
git add .

# Commit with meaningful message
git commit -m "feat(generation): implement reflexion loop for multi-agent system"
```

**Commit Message Convention:**
```
<type>(<area>): <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- test: Tests
- refactor: Code refactoring
- chore: Maintenance

Areas:
- generation
- evaluation
- taxonomy
- multimodal
- shared
- docs
```

### **Step 5: Push Your Branch**

```bash
# Push your branch to GitHub
git push origin generation/reflexion-agent  # your branch name
```

### **Step 6: Create a Pull Request (PR)**

1. Go to GitHub repository
2. Click "Compare & pull request"
3. **Title:** Clear description of what you did
4. **Description:** 
   - What changes were made
   - Which files were affected
   - Any dependencies or requirements
5. **Assign Reviewers:** Tag relevant team members
6. Click "Create pull request"

### **Step 7: Code Review & Merge**

- Team members review your code
- Address any feedback
- Once approved → Merge to `main`
- Delete your feature branch after merge

---

## 🔄 Keeping Your Branch Updated

If `main` branch has new changes while you're working:

```bash
# Get latest changes from main
git checkout main
git pull origin main

# Switch back to your branch
git checkout your-branch-name

# Merge main into your branch (or rebase)
git merge main

# Resolve any conflicts if they occur
# Then push updated branch
git push origin your-branch-name
```

---

## 🚨 Handling Merge Conflicts

Conflicts happen when two people edit the same file. **With our folder structure, this should be RARE** because each person works in different folders.

If you get a conflict:

```bash
# Git will mark conflicted files
git status

# Open the conflicted file
# Look for markers like:
<<<<<<< HEAD
Your changes
=======
Their changes
>>>>>>> main

# Edit the file to resolve the conflict
# Remove the markers
# Keep the correct version

# Mark as resolved
git add conflicted_file.py

# Complete the merge
git commit -m "merge: resolve conflict in conflicted_file.py"
```

---

## 🤝 Collaboration Best Practices

### ✅ DO:

1. **Work in Your Folder** - Stay in your designated directory
   - `generation/multi-agent/` for Aziz
   - `evaluation/` for evaluation researcher
   - `taxonomy/` for taxonomy researcher
   - `multimodal/` for multimodal researcher

2. **Use Feature Branches** - Always branch off `main`
   ```bash
   git checkout -b feature-name
   ```

3. **Pull Before Push** - Always update before pushing
   ```bash
   git pull origin main
   ```

4. **Write Good Commit Messages**
   ```bash
   git commit -m "feat(generation): add HyDE query expansion"
   ```

5. **Use Shared Resources** - Put reusable code in `shared/`
   ```python
   from shared.utils.llm import OpenAIClient
   ```

6. **Document Your Work** - Update READMEs in your folder

7. **Small, Frequent Commits** - Commit often with small changes

8. **Review Others' PRs** - Help review code from other tracks

### ❌ DON'T:

1. **Don't Commit to Main Directly** - Always use branches

2. **Don't Edit Other People's Folders** - Ask first or coordinate

3. **Don't Commit Large Files** - Use `.gitignore`
   - PDFs > 50MB
   - Model weights
   - Large datasets

4. **Don't Commit API Keys** - Use `.env` files (in `.gitignore`)

5. **Don't Merge Your Own PRs** - Wait for review

6. **Don't Force Push** - Unless you're sure
   ```bash
   # Dangerous! Only if you know what you're doing
   git push --force
   ```

---

## 📊 Example Workflow: Aziz Working on Multi-Agent

```bash
# 1. Start fresh
cd c:\Users\benam\Downloads\Agentic_AI
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b generation/implement-critic-agent

# 3. Work on your feature
cd generation/multi-agent/src/agents/
# ... create critic_agent.py ...

# 4. Test your changes
pytest generation/multi-agent/tests/

# 5. Commit
git add generation/multi-agent/src/agents/critic_agent.py
git add generation/multi-agent/tests/test_critic.py
git commit -m "feat(generation): implement Critic agent with Constitutional AI"

# 6. Push
git push origin generation/implement-critic-agent

# 7. Create PR on GitHub
# 8. Wait for review
# 9. Merge to main
# 10. Delete branch and start next feature
```

---

## 🔧 Common Git Commands

```bash
# Check status
git status

# See what changed
git diff

# View commit history
git log --oneline

# Switch branches
git checkout branch-name

# Create new branch
git checkout -b new-branch-name

# Update from remote
git pull origin main

# Stash changes (temporary save)
git stash
git stash pop  # restore

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all changes (DANGEROUS!)
git reset --hard HEAD
```

---

## 🎯 Milestones & Integration

### Month 1 (Weeks 1-4)
- Each track: Literature review & planning
- **Sync:** Share findings in `docs/literature-review/`

### Month 2 (Weeks 5-8)
- Each track: Initial implementation
- **Sync:** Share utilities in `shared/utils/`

### Month 3 (Weeks 9-12)
- Each track: Complete implementation
- **Sync:** Cross-track testing begins

### Month 4 (Weeks 13-16)
- **Integration:** Combine approaches
- **Benchmarking:** Compare methods
- **Publication:** Prepare HuggingFace dataset

---

## 📞 Communication Channels

1. **GitHub Issues** - For bugs, questions, tasks
2. **Pull Requests** - For code review and discussion
3. **Weekly Sync** - Team meeting to coordinate
4. **Slack/Discord** - Quick questions
5. **This README** - Documentation

---

## 🆘 Getting Help

### "I have a merge conflict!"
→ See "Handling Merge Conflicts" section above

### "I accidentally committed to main!"
```bash
git reset --soft HEAD~1  # Undo commit, keep changes
git checkout -b feature-name  # Create proper branch
git push origin feature-name
```

### "I need to use someone else's code!"
→ Use `shared/` directory or import from their folder:
```python
from generation.multi_agent.src.utils import some_function
```

### "My branch is behind main!"
```bash
git checkout main
git pull
git checkout your-branch
git merge main
```

---

## 📚 Resources

- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Pull Request Best Practices](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/)

---

