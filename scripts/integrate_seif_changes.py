"""
Quick Integration Script - Merge Seif's Changes
================================================

This script helps you safely integrate Seif's improvements:
1. Backup current critic_agent.py
2. Copy Seif's version
3. Copy new validators
4. Update requirements.txt
5. Run verification tests

Usage:
    python integrate_seif_changes.py --backup --copy --test
"""

import shutil
import os
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent
SEIF_REVIEW = PROJECT_ROOT / "seif_changes_review"
SRC_AGENTS = PROJECT_ROOT / "src" / "agents"
SRC_UTILS = PROJECT_ROOT / "src" / "utils"
BACKUP_DIR = PROJECT_ROOT / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

def create_backup():
    """Backup current critic_agent.py"""
    print("=" * 80)
    print("STEP 1: Creating backup...")
    print("=" * 80)
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Backup critic_agent.py
    source = SRC_AGENTS / "critic_agent.py"
    dest = BACKUP_DIR / "critic_agent.py"
    shutil.copy2(source, dest)
    print(f"✅ Backed up: {source} → {dest}")
    
    return True

def copy_seif_files():
    """Copy Seif's files to project"""
    print("\n" + "=" * 80)
    print("STEP 2: Copying Seif's files...")
    print("=" * 80)
    
    files_to_copy = [
        # (source, destination)
        (SEIF_REVIEW / "critic_agent_seif.py", SRC_AGENTS / "critic_agent.py"),
        (SEIF_REVIEW / "answer_quality_scorer.py", SRC_AGENTS / "answer_quality_scorer.py"),
        (SEIF_REVIEW / "chain_of_thought_validator.py", SRC_AGENTS / "chain_of_thought_validator.py"),
        (SEIF_REVIEW / "active_learning_ui.py", SRC_UTILS / "active_learning_ui.py"),
    ]
    
    for source, dest in files_to_copy:
        if source.exists():
            shutil.copy2(source, dest)
            print(f"✅ Copied: {source.name} → {dest}")
        else:
            print(f"⚠️  Not found: {source}")
    
    return True

def update_requirements():
    """Add new dependencies to requirements.txt"""
    print("\n" + "=" * 80)
    print("STEP 3: Updating requirements.txt...")
    print("=" * 80)
    
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    new_deps = [
        "gradio>=4.0.0  # For active learning UI",
        "spacy>=3.7.0  # Optional: NER for AnswerQualityScorer"
    ]
    
    # Read existing requirements
    with open(requirements_file, 'r') as f:
        existing = f.read()
    
    # Add new dependencies if not present
    for dep in new_deps:
        dep_name = dep.split('>=')[0]
        if dep_name not in existing:
            with open(requirements_file, 'a') as f:
                f.write(f"\n{dep}")
            print(f"✅ Added: {dep}")
        else:
            print(f"⏭️  Already present: {dep_name}")
    
    return True

def run_tests():
    """Run verification tests"""
    print("\n" + "=" * 80)
    print("STEP 4: Running verification tests...")
    print("=" * 80)
    
    import subprocess
    import sys
    
    tests = [
        ("Test validators", "test_seif_validators.py"),
        ("Test pipeline", "test_pipeline_local.py"),
    ]
    
    for test_name, test_file in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 80)
        
        try:
            # Try to find python executable
            python_cmd = sys.executable
            result = subprocess.run(
                [python_cmd, test_file],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                print(result.stdout)
                print(result.stderr)
        
        except FileNotFoundError:
            print(f"⚠️  Python not found, please run manually: python {test_file}")
        except subprocess.TimeoutExpired:
            print(f"⏱️  {test_name} timed out (>5min), check manually")
        except Exception as e:
            print(f"⚠️  Error running {test_name}: {e}")
    
    return True

def print_summary():
    """Print integration summary"""
    print("\n" + "=" * 80)
    print("✅ INTEGRATION COMPLETE")
    print("=" * 80)
    print("\n📊 What Changed:")
    print("  1. critic_agent.py → Adversarial prompting + 5 hard rules")
    print("  2. answer_quality_scorer.py → NEW (hallucination detector)")
    print("  3. chain_of_thought_validator.py → NEW (reasoning validator)")
    print("  4. active_learning_ui.py → NEW (human review interface)")
    print("\n📈 Expected Results:")
    print("  - Rejection rate: 15% → 33%")
    print("  - Score distribution: 0.94-1.00 → 0.50-1.00 (varied)")
    print("  - Retry loops: 1-2 → 15-20 per run")
    print("\n🚀 Next Steps:")
    print("  1. Run: python test_pipeline_local.py")
    print("  2. Check rejection rate (should be ~30-35%)")
    print("  3. Review rejection reasons in logs")
    print("  4. Plan validator integration into pipeline.py")
    print("\n💡 Rollback (if needed):")
    print(f"  cp {BACKUP_DIR}/critic_agent.py src/agents/critic_agent.py")
    print("\n📖 Read: SEIF_CHANGES_ANALYSIS.md for full details")

def main():
    """Main integration workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate Seif's changes")
    parser.add_argument('--backup', action='store_true', help='Create backup')
    parser.add_argument('--copy', action='store_true', help='Copy files')
    parser.add_argument('--requirements', action='store_true', help='Update requirements.txt')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--all', action='store_true', help='Do everything')
    
    args = parser.parse_args()
    
    if not any([args.backup, args.copy, args.requirements, args.test, args.all]):
        print("Usage: python integrate_seif_changes.py --all")
        print("   or: python integrate_seif_changes.py --backup --copy --test")
        return
    
    try:
        if args.all or args.backup:
            create_backup()
        
        if args.all or args.copy:
            copy_seif_files()
        
        if args.all or args.requirements:
            update_requirements()
        
        if args.all or args.test:
            run_tests()
        
        print_summary()
        
    except Exception as e:
        print(f"\n❌ Error during integration: {e}")
        print(f"Backup available at: {BACKUP_DIR}")
        raise

if __name__ == "__main__":
    main()
