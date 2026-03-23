"""
Active Learning Loop - Human-in-the-Loop Quality Improvement
=============================================================

Purpose:
--------
Gradio-based UI for human review and continuous improvement of generated datasets.

Features:
---------
1. Review Interface: Accept/Reject/Edit QA pairs
2. Quality Insights: Show scores from all validators
3. Batch Review: Review multiple entries efficiently
4. Feedback Collection: Capture why humans reject/edit
5. Statistics Dashboard: Track acceptance rate, common issues
6. Export Reviewed: Save human-validated dataset

Workflow:
---------
1. Generate dataset with pipeline
2. Load into Active Learning UI
3. Human reviews samples, provides feedback
4. System learns from feedback patterns
5. Export high-quality human-validated dataset

Usage:
------
python active_learning_ui.py --dataset output/dataset.json

Or from code:
>>> from active_learning_ui import launch_review_ui
>>> launch_review_ui("output/dataset.json")

Author: Seif
Date: 2025
"""

import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd


@dataclass
class ReviewDecision:
    """Human review decision for a QA pair."""
    entry_id: str
    decision: str  # "accept", "reject", "edit"
    edited_question: Optional[str] = None
    edited_answer: Optional[str] = None
    feedback: Optional[str] = None  # Why rejected/edited
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ActiveLearningSession:
    """
    Manages an active learning review session.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize review session.
        
        Args:
            dataset_path: Path to dataset JSON file
        """
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset(dataset_path)
        self.current_index = 0
        self.reviews: List[ReviewDecision] = []
        self.stats = {
            'total': len(self.dataset),
            'reviewed': 0,
            'accepted': 0,
            'rejected': 0,
            'edited': 0
        }
    
    def _load_dataset(self, path: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats (with metadata or just data array)
        if 'data' in data:
            return data['data']
        else:
            return data
    
    def get_current_entry(self) -> Optional[Dict]:
        """Get current entry for review."""
        if self.current_index < len(self.dataset):
            entry = self.dataset[self.current_index].copy()
            entry['entry_index'] = self.current_index
            return entry
        return None
    
    def submit_review(
        self,
        decision: str,
        edited_question: Optional[str] = None,
        edited_answer: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Submit review for current entry.
        
        Args:
            decision: "accept", "reject", or "edit"
            edited_question: Edited question (if decision="edit")
            edited_answer: Edited answer (if decision="edit")
            feedback: Explanation for decision
            
        Returns:
            True if review submitted successfully
        """
        current = self.get_current_entry()
        if not current:
            return False
        
        # Create review
        review = ReviewDecision(
            entry_id=current.get('chunk_id', str(self.current_index)),
            decision=decision,
            edited_question=edited_question if decision == "edit" else None,
            edited_answer=edited_answer if decision == "edit" else None,
            feedback=feedback
        )
        
        self.reviews.append(review)
        
        # Update stats
        self.stats['reviewed'] += 1
        if decision == "accept":
            self.stats['accepted'] += 1
        elif decision == "reject":
            self.stats['rejected'] += 1
        elif decision == "edit":
            self.stats['edited'] += 1
        
        # Move to next
        self.current_index += 1
        
        return True
    
    def skip_entry(self) -> bool:
        """Skip current entry without review."""
        if self.current_index < len(self.dataset):
            self.current_index += 1
            return True
        return False
    
    def go_back(self) -> bool:
        """Go back to previous entry."""
        if self.current_index > 0:
            self.current_index -= 1
            # Remove last review if exists
            if self.reviews and self.reviews[-1].entry_id == str(self.current_index):
                last_review = self.reviews.pop()
                # Update stats
                self.stats['reviewed'] -= 1
                if last_review.decision == "accept":
                    self.stats['accepted'] -= 1
                elif last_review.decision == "reject":
                    self.stats['rejected'] -= 1
                elif last_review.decision == "edit":
                    self.stats['edited'] -= 1
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get current review statistics."""
        stats = self.stats.copy()
        if stats['reviewed'] > 0:
            stats['acceptance_rate'] = stats['accepted'] / stats['reviewed'] * 100
            stats['rejection_rate'] = stats['rejected'] / stats['reviewed'] * 100
            stats['edit_rate'] = stats['edited'] / stats['reviewed'] * 100
            stats['progress'] = stats['reviewed'] / stats['total'] * 100
        else:
            stats['acceptance_rate'] = 0
            stats['rejection_rate'] = 0
            stats['edit_rate'] = 0
            stats['progress'] = 0
        
        return stats
    
    def export_reviews(self, output_path: str):
        """Export reviews to JSON file."""
        output = {
            'dataset_source': self.dataset_path,
            'review_date': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'reviews': [r.to_dict() for r in self.reviews]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def export_accepted_dataset(self, output_path: str):
        """Export dataset with only accepted/edited entries."""
        accepted_entries = []
        
        for review in self.reviews:
            if review.decision in ["accept", "edit"]:
                # Find original entry
                entry_id = review.entry_id
                original = next(
                    (e for e in self.dataset if e.get('chunk_id') == entry_id),
                    None
                )
                
                if original:
                    entry = original.copy()
                    
                    # Apply edits if present
                    if review.decision == "edit":
                        if review.edited_question:
                            entry['question'] = review.edited_question
                        if review.edited_answer:
                            entry['answer'] = review.edited_answer
                        entry['human_edited'] = True
                    else:
                        entry['human_edited'] = False
                    
                    entry['human_reviewed'] = True
                    accepted_entries.append(entry)
        
        # Create output with metadata
        output = {
            'metadata': {
                'source_dataset': self.dataset_path,
                'review_date': datetime.now().isoformat(),
                'total_entries': len(accepted_entries),
                'acceptance_rate': self.get_statistics()['acceptance_rate'],
                'human_validated': True
            },
            'data': accepted_entries
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


def create_gradio_interface(session: ActiveLearningSession):
    """
    Create Gradio interface for active learning review.
    
    Args:
        session: ActiveLearningSession instance
        
    Returns:
        Gradio Blocks interface
    """
    
    def load_entry():
        """Load current entry for display."""
        entry = session.get_current_entry()
        if not entry:
            return (
                "✅ Review Complete!",
                "",
                "",
                "",
                "",
                format_stats(),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False)
            )
        
        # Format metadata
        metadata = f"""
**Source**: {entry.get('source_file', 'N/A')}
**Chunk ID**: {entry.get('chunk_id', 'N/A')}
**Type**: {entry.get('question_type', 'N/A')} | **Difficulty**: {entry.get('difficulty', 'N/A')}
**Quality Score**: {entry.get('critic_score', 0):.2f}
**Progress**: {session.current_index + 1}/{len(session.dataset)}
        """
        
        question = entry.get('question', '')
        answer = entry.get('answer', '')
        chunk_preview = entry.get('chunk_content', '')[:500] + "..."
        
        return (
            metadata,
            question,
            answer,
            chunk_preview,
            "",  # Clear feedback
            format_stats(),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
    
    def format_stats():
        """Format statistics for display."""
        stats = session.get_statistics()
        return f"""
### 📊 Review Statistics
- **Progress**: {stats['progress']:.1f}% ({stats['reviewed']}/{stats['total']})
- **Accepted**: {stats['accepted']} ({stats['acceptance_rate']:.1f}%)
- **Rejected**: {stats['rejected']} ({stats['rejection_rate']:.1f}%)
- **Edited**: {stats['edited']} ({stats['edit_rate']:.1f}%)
        """
    
    def accept_entry(feedback_text):
        """Accept current entry."""
        session.submit_review("accept", feedback=feedback_text)
        return load_entry()
    
    def reject_entry(feedback_text):
        """Reject current entry."""
        if not feedback_text:
            return gr.update(), gr.update(), gr.update(), gr.update(), \
                   "⚠️ Please provide feedback explaining why you rejected this entry.", \
                   gr.update(), gr.update(), gr.update(), gr.update()
        
        session.submit_review("reject", feedback=feedback_text)
        return load_entry()
    
    def edit_entry(question_text, answer_text, feedback_text):
        """Edit and accept current entry."""
        session.submit_review(
            "edit",
            edited_question=question_text,
            edited_answer=answer_text,
            feedback=feedback_text
        )
        return load_entry()
    
    def skip():
        """Skip current entry."""
        session.skip_entry()
        return load_entry()
    
    def go_back():
        """Go back to previous entry."""
        session.go_back()
        return load_entry()
    
    def export_reviews():
        """Export review decisions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reviews_{timestamp}.json"
        session.export_reviews(output_path)
        return f"✅ Reviews exported to: {output_path}"
    
    def export_dataset():
        """Export accepted dataset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"dataset_reviewed_{timestamp}.json"
        session.export_accepted_dataset(output_path)
        return f"✅ Reviewed dataset exported to: {output_path}"
    
    # Create Gradio interface
    with gr.Blocks(title="Active Learning Review") as interface:
        gr.Markdown("# 🔍 Active Learning Review Interface")
        gr.Markdown("Review generated QA pairs and provide feedback to improve dataset quality.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Entry display
                metadata_box = gr.Markdown("Loading...")
                
                gr.Markdown("### 📝 Question")
                question_box = gr.Textbox(
                    label="Question",
                    lines=3,
                    interactive=True,
                    placeholder="Question will appear here..."
                )
                
                gr.Markdown("### 💬 Answer")
                answer_box = gr.Textbox(
                    label="Answer",
                    lines=6,
                    interactive=True,
                    placeholder="Answer will appear here..."
                )
                
                gr.Markdown("### 📄 Source Context (Preview)")
                chunk_box = gr.Textbox(
                    label="Chunk Content",
                    lines=4,
                    interactive=False,
                    placeholder="Chunk preview will appear here..."
                )
                
                gr.Markdown("### 💭 Feedback (Optional)")
                feedback_box = gr.Textbox(
                    label="Why accept/reject/edit?",
                    lines=2,
                    placeholder="Explain your decision..."
                )
            
            with gr.Column(scale=1):
                # Statistics
                stats_box = gr.Markdown(format_stats())
                
                gr.Markdown("### 🎯 Actions")
                
                accept_btn = gr.Button("✅ Accept", variant="primary")
                reject_btn = gr.Button("❌ Reject", variant="stop")
                edit_btn = gr.Button("✏️ Edit & Accept", variant="secondary")
                
                gr.Markdown("---")
                
                skip_btn = gr.Button("⏭️ Skip")
                back_btn = gr.Button("⬅️ Go Back")
                
                gr.Markdown("### 💾 Export")
                
                export_reviews_btn = gr.Button("Export Reviews (JSON)")
                export_dataset_btn = gr.Button("Export Reviewed Dataset")
                
                export_status = gr.Textbox(label="Export Status", lines=2)
        
        # Event handlers
        interface.load(
            fn=load_entry,
            inputs=[],
            outputs=[
                metadata_box, question_box, answer_box, chunk_box,
                feedback_box, stats_box,
                accept_btn, reject_btn, edit_btn
            ]
        )
        
        accept_btn.click(
            fn=accept_entry,
            inputs=[feedback_box],
            outputs=[
                metadata_box, question_box, answer_box, chunk_box,
                feedback_box, stats_box,
                accept_btn, reject_btn, edit_btn
            ]
        )
        
        reject_btn.click(
            fn=reject_entry,
            inputs=[feedback_box],
            outputs=[
                metadata_box, question_box, answer_box, chunk_box,
                feedback_box, stats_box,
                accept_btn, reject_btn, edit_btn
            ]
        )
        
        edit_btn.click(
            fn=edit_entry,
            inputs=[question_box, answer_box, feedback_box],
            outputs=[
                metadata_box, question_box, answer_box, chunk_box,
                feedback_box, stats_box,
                accept_btn, reject_btn, edit_btn
            ]
        )
        
        skip_btn.click(
            fn=skip,
            inputs=[],
            outputs=[
                metadata_box, question_box, answer_box, chunk_box,
                feedback_box, stats_box,
                accept_btn, reject_btn, edit_btn
            ]
        )
        
        back_btn.click(
            fn=go_back,
            inputs=[],
            outputs=[
                metadata_box, question_box, answer_box, chunk_box,
                feedback_box, stats_box,
                accept_btn, reject_btn, edit_btn
            ]
        )
        
        export_reviews_btn.click(
            fn=export_reviews,
            inputs=[],
            outputs=[export_status]
        )
        
        export_dataset_btn.click(
            fn=export_dataset,
            inputs=[],
            outputs=[export_status]
        )
    
    return interface


def launch_review_ui(dataset_path: str, share: bool = False):
    """
    Launch Gradio review UI.
    
    Args:
        dataset_path: Path to dataset JSON file
        share: Create shareable link
    """
    print(f"🚀 Launching Active Learning Review UI...")
    print(f"📁 Dataset: {dataset_path}")
    
    session = ActiveLearningSession(dataset_path)
    interface = create_gradio_interface(session)
    
    interface.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True  # Auto-open browser
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Active Learning Review UI")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to dataset JSON file",
        nargs="?",
        default="output/dataset.json"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create shareable Gradio link"
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("\nPlease provide a valid dataset path:")
        print("  python active_learning_ui.py path/to/dataset.json")
        exit(1)
    
    launch_review_ui(args.dataset, share=args.share)
