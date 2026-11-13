import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ioh

from analysis.plot_search_res import plot_search_result


def run_stage1_plotter(out_base: Path):
    targets = {
        'bbob_f8': Path('outputs/bbob_f8'),
        'real_world_1124': Path('outputs/real_world_1124'),
    }

    for target_name, target_dir in targets.items():
        if not target_dir.exists():
            print(f'Directory not found: {target_dir}')
            continue

        subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print(f'No population directories found in {target_dir}')
            continue

        for sub in subdirs:
            final_files = list(sub.glob('*final*.pkl'))
            if not final_files:
                all_pkls = sorted(sub.glob('*.pkl'), key=lambda p: p.stat().st_mtime)
                if all_pkls:
                    fallback_final = sub / f'{all_pkls[-1].stem}_final.pkl'
                    shutil.copy2(all_pkls[-1], fallback_final)
                    print(f'Created fallback final checkpoint: {fallback_final.name}')

        fig_dir = out_base / f'stage1_{target_name}'
        fig_dir.mkdir(parents=True, exist_ok=True)
        cache_pkl = fig_dir / f'cached_{target_name}_df.pkl'

        print(f"Running plot_search_result on '{target_dir}' -> '{fig_dir}'")
        plot_search_result(
            result_dir=str(target_dir),
            save_name=str(cache_pkl),
            extract_fn=None,
            fig_dir=str(fig_dir),
        )
        print(f'Stage 1 plots saved in: {fig_dir} (es_aoc_lines.png, search_aoc_voilin.png)')


def plot_stage2_convergence(out_base: Path):
    candidates = [
        Path('stage2_convergence_data_combined.csv'),
        Path('outputs/stage2_benchmark/stage2_convergence_data_combined.csv'),
    ]
    conv_path = next((p for p in candidates if p.exists()), None)
    if conv_path is None:
        print('stage2_convergence_data_combined.csv not found.')
        return

    df_raw = pd.read_csv(conv_path)
    bbob_f_opt = float(ioh.get_problem(8, instance=1, dimension=6).optimum.y)

    algo_order = [
        'Generated_BBOB (AdaptiveTrustHybridBO)',
        'Generated_RealWorld (AdaptiveTrustRegionBO)',
        'BoTorch_Vanilla',
        'CMA_ES',
    ]
    display_names = {
        'Generated_BBOB (AdaptiveTrustHybridBO)': 'Generated BBOB (AdaptiveTrustHybridBO)',
        'Generated_RealWorld (AdaptiveTrustRegionBO)': 'Generated Real-World (AdaptiveTrustRegionBO)',
        'BoTorch_Vanilla': 'BoTorch Vanilla (GP + EI)',
        'CMA_ES': 'CMA-ES',
    }
    styles = {
        'Generated_BBOB (AdaptiveTrustHybridBO)': ('#1f77b4', '-', 2.2),
        'Generated_RealWorld (AdaptiveTrustRegionBO)': ('#ff7f0e', '--', 2.2),
        'BoTorch_Vanilla': ('#2ca02c', '-.', 2.0),
        'CMA_ES': ('#d62728', ':', 2.2),
    }

    specs = [
        {
            'problem': 'BBOB_F8_Rosenbrock',
            'title': 'Stage 2 Convergence: COCO-BBOB F8 (Rosenbrock, d=6)',
            'ylabel': 'Loss: Best-So-Far Regret $f(x) - f_{opt}$ (Mean ± SE)',
            'offset': bbob_f_opt,
            'stem': 'stage2_bbob_f8_loss_convergence',
        },
        {
            'problem': 'RealWorld_1124_FMSound',
            'title': 'Stage 2 Convergence: Practical Problem 1124 (FM Sound Waves, d=6)',
            'ylabel': 'Best-So-Far Objective Value (Mean ± SE)',
            'offset': 0.0,
            'stem': 'stage2_realworld_1124_objective_convergence',
        },
    ]

    print('\nGenerating stage 2 plots.')
    for spec in specs:
        sub = df_raw[df_raw['Problem'] == spec['problem']].copy()
        if sub.empty:
            continue

        sub['Metric'] = np.maximum(sub['Best_So_Far'] - spec['offset'], 1e-12)
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

        for algo in algo_order:
            algo_df = sub[sub['Algorithm'] == algo]
            if algo_df.empty:
                continue

            pivot = algo_df.pivot(index='Evaluation', columns='Seed', values='Metric')
            evals = pivot.index.values
            mean_vals = pivot.mean(axis=1).values
            se_vals = pivot.std(axis=1, ddof=1).values / np.sqrt(pivot.shape[1])

            color, ls, lw = styles[algo]
            ax.plot(evals, mean_vals, label=display_names[algo], color=color, linestyle=ls, linewidth=lw)
            ax.fill_between(
                evals,
                np.maximum(mean_vals - se_vals, 1e-12),
                mean_vals + se_vals,
                color=color,
                alpha=0.18,
            )

        ax.set_title(spec['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Function Evaluations (Budget = 600)', fontsize=11)
        ax.set_ylabel(spec['ylabel'], fontsize=11)
        ax.set_yscale('log')
        ax.set_xlim(1, int(sub['Evaluation'].max()))
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
        ax.legend(fontsize=9.5, loc='upper right', framealpha=0.92)

        plt.tight_layout()
        png_path = out_base / f"{spec['stem']}.png"
        pdf_path = out_base / f"{spec['stem']}.pdf"
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)
        plt.close()
        print(f'Saved plot: {png_path}')


def main():
    out_base = Path('outputs/convergence_analysis')
    out_base.mkdir(parents=True, exist_ok=True)
    run_stage1_plotter(out_base)
    plot_stage2_convergence(out_base)
    print('\nAll plots generated in:', out_base)


if __name__ == '__main__':
    main()