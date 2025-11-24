"""
Generate social media graphics for Credit Card Fraud Detection project
Creates LinkedIn and Kaggle promotional graphics without performance metrics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Create output directory
output_dir = Path("social_media")
output_dir.mkdir(exist_ok=True)

# Color scheme - Financial security theme
PRIMARY_COLOR = "#1a472a"  # Dark green
ACCENT_COLOR = "#2ecc71"   # Bright green
DANGER_COLOR = "#e74c3c"   # Red
BACKGROUND = "#0f1419"     # Dark background
TEXT_COLOR = "#ffffff"     # White text

def create_linkedin_post():
    """Create LinkedIn post graphic (1200x627px)"""
    fig, ax = plt.subplots(figsize=(12, 6.27), facecolor=BACKGROUND)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.27)
    ax.axis('off')

    # Background gradient effect
    rect = patches.Rectangle((0, 0), 12, 6.27, linewidth=0,
                             edgecolor='none', facecolor=BACKGROUND)
    ax.add_patch(rect)

    # Accent stripe
    accent_rect = patches.Rectangle((0, 0), 12, 1.5, linewidth=0,
                                   edgecolor='none', facecolor=PRIMARY_COLOR, alpha=0.8)
    ax.add_patch(accent_rect)

    # Shield icon representation (security symbol)
    shield = patches.FancyBboxPatch((0.5, 2.5), 3, 3,
                                   boxstyle="round,pad=0.1",
                                   linewidth=3, edgecolor=ACCENT_COLOR,
                                   facecolor=PRIMARY_COLOR, alpha=0.9)
    ax.add_patch(shield)

    # Lock symbol inside shield
    lock_body = patches.Rectangle((1.5, 3), 1, 1.2, linewidth=0,
                                 edgecolor='none', facecolor=ACCENT_COLOR)
    ax.add_patch(lock_body)

    lock_top = patches.Wedge((2, 4.2), 0.4, 0, 180, linewidth=0,
                            edgecolor='none', facecolor=ACCENT_COLOR)
    ax.add_patch(lock_top)

    # Main title
    ax.text(5, 5, 'FraudGuard AI', fontsize=56, weight='bold',
            color=ACCENT_COLOR, ha='left', va='center',
            fontfamily='sans-serif')

    # Subtitle
    ax.text(5, 4, 'Real-Time Credit Card Fraud Detection', fontsize=28,
            color=TEXT_COLOR, ha='left', va='center', alpha=0.9,
            fontfamily='sans-serif')

    # Technology badges
    technologies = ['XGBoost', 'SHAP', 'FastAPI', 'Streamlit']
    x_start = 5
    y_pos = 2.8

    for i, tech in enumerate(technologies):
        tech_rect = patches.FancyBboxPatch((x_start + i*1.6, y_pos - 0.2), 1.4, 0.4,
                                          boxstyle="round,pad=0.05",
                                          linewidth=2, edgecolor=ACCENT_COLOR,
                                          facecolor=PRIMARY_COLOR, alpha=0.7)
        ax.add_patch(tech_rect)
        ax.text(x_start + i*1.6 + 0.7, y_pos, tech, fontsize=14,
               color=TEXT_COLOR, ha='center', va='center', weight='bold',
               fontfamily='sans-serif')

    # Features
    features = [
        '• Explainable AI with SHAP',
        '• Production-Ready API',
        '• Interactive Dashboard',
        '• SMOTE for Imbalanced Data'
    ]

    y_feature = 2
    for i, feature in enumerate(features):
        ax.text(5, y_feature - i*0.35, feature, fontsize=16,
               color=TEXT_COLOR, ha='left', va='center', alpha=0.85,
               fontfamily='sans-serif')

    # Footer
    ax.text(6, 0.4, 'Open Source • MIT License', fontsize=14,
           color=TEXT_COLOR, ha='center', va='center', alpha=0.6,
           fontfamily='sans-serif', style='italic')

    plt.tight_layout(pad=0)
    plt.savefig(output_dir / 'linkedin_post.png', dpi=100, facecolor=BACKGROUND,
               bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[OK] Created LinkedIn post: {output_dir / 'linkedin_post.png'}")

def create_kaggle_thumbnail():
    """Create Kaggle thumbnail graphic (640x512px)"""
    fig, ax = plt.subplots(figsize=(6.4, 5.12), facecolor=BACKGROUND)
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 5.12)
    ax.axis('off')

    # Background
    rect = patches.Rectangle((0, 0), 6.4, 5.12, linewidth=0,
                            edgecolor='none', facecolor=BACKGROUND)
    ax.add_patch(rect)

    # Top accent bar
    accent_rect = patches.Rectangle((0, 4), 6.4, 1.12, linewidth=0,
                                   edgecolor='none', facecolor=PRIMARY_COLOR, alpha=0.9)
    ax.add_patch(accent_rect)

    # Credit card icon
    card = patches.FancyBboxPatch((0.5, 2.2), 2, 1.3,
                                 boxstyle="round,pad=0.05",
                                 linewidth=3, edgecolor=ACCENT_COLOR,
                                 facecolor=PRIMARY_COLOR, alpha=0.9)
    ax.add_patch(card)

    # Card chip
    chip = patches.Rectangle((0.7, 3), 0.4, 0.3, linewidth=0,
                            edgecolor='none', facecolor=ACCENT_COLOR)
    ax.add_patch(chip)

    # Warning symbol
    warning = patches.RegularPolygon((4.5, 2.85), 3, radius=0.5, linewidth=3,
                                    edgecolor=DANGER_COLOR, facecolor=BACKGROUND,
                                    orientation=0)
    ax.add_patch(warning)
    ax.text(4.5, 2.85, '!', fontsize=36, weight='bold',
           color=DANGER_COLOR, ha='center', va='center')

    # Title
    ax.text(3.2, 4.5, 'FraudGuard', fontsize=36, weight='bold',
           color=ACCENT_COLOR, ha='center', va='center',
           fontfamily='sans-serif')

    # Subtitle
    ax.text(3.2, 1.5, 'AI-Powered', fontsize=24, weight='bold',
           color=TEXT_COLOR, ha='center', va='center',
           fontfamily='sans-serif')

    ax.text(3.2, 1, 'Fraud Detection', fontsize=24, weight='bold',
           color=TEXT_COLOR, ha='center', va='center',
           fontfamily='sans-serif')

    # Tech stack
    ax.text(3.2, 0.4, 'XGBoost • SHAP • FastAPI', fontsize=14,
           color=TEXT_COLOR, ha='center', va='center', alpha=0.7,
           fontfamily='sans-serif')

    plt.tight_layout(pad=0)
    plt.savefig(output_dir / 'kaggle_thumbnail.png', dpi=100, facecolor=BACKGROUND,
               bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[OK] Created Kaggle thumbnail: {output_dir / 'kaggle_thumbnail.png'}")

def main():
    """Generate all social media graphics"""
    print("Generating social media graphics...")
    print("-" * 50)

    create_linkedin_post()
    create_kaggle_thumbnail()

    print("-" * 50)
    print(f"[OK] All graphics saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  - linkedin_post.png (1200x627px)")
    print(f"  - kaggle_thumbnail.png (640x512px)")

if __name__ == "__main__":
    main()
