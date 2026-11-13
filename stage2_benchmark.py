import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import random
import inspect
import importlib.util
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
import cma
import ioh
from Practical_Problems.Practical_Problems_Suite import get_practical_problem
from baseline.botorch_vanilla import Vanilla_BO

from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


torch.set_num_threads(1)
MAX_WORKERS = 6
RUN_TIMEOUTS = {'custom': 600, 'botorch': 900, 'cmaes': 120}

def _run_one_task(task):
    prob_label, algo_label, algo_type, algo_path, prob_id, inst_id, seed, budget, dim = task

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    problem = get_target_problem(prob_id, inst_id, dim)

    if algo_type == 'custom':
        algo_cls, _ = load_generated_class(Path(algo_path))   # reload in worker
        curve = run_generated_bo(algo_cls, problem, budget, dim, seed)
    elif algo_type == 'botorch':
        curve = run_botorch_vanilla(problem, budget, dim, seed)
    elif algo_type == 'cmaes':
        curve = run_cma_es(problem, budget, dim, seed)
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")

    return prob_label, algo_label, seed, curve

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path='configs.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    group_id = cfg['experiment']['group_id']
    dim = cfg['experiment']['dimension']
    budget = cfg['stage2_generalization']['evaluation_budget']
    configured_seeds = cfg['experiment']['seeds']

    expected_seeds = [10 * group_id + r for r in range(10)]
    assert configured_seeds == expected_seeds, (f'Seed mismatch found. For Group {group_id}, expected {expected_seeds}, 'f'but configs.yaml contains {configured_seeds}')
    return cfg, configured_seeds, dim, budget

def load_generated_class(py_file_path: Path):
    spec = importlib.util.spec_from_file_location(py_file_path.stem, py_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        if attr.endswith('BO') and isinstance(getattr(mod, attr), type):
            return getattr(mod, attr), attr
    raise AttributeError(f'No BO class found in {py_file_path}')

def get_target_problem(problem_id: int, instance: int, dim: int):
    if problem_id >= 1000:
        prob = get_practical_problem(problem_id, dim)
    else:
        prob = ioh.get_problem(problem_id, instance, dim)
    prob.reset()
    return prob

# 1. Generated BO algorithms
def run_generated_bo(algo_cls, problem, budget: int, dim: int, seed: int) -> np.array:
    set_all_seeds(seed)

    eval_count = 0
    best_so_far = []
    current_best = float('inf')

    def tracked_func(x):
        nonlocal current_best, eval_count
        if eval_count >= budget:
            return current_best
        val = problem(np.asarray(x, dtype=np.float64).ravel())
        eval_count += 1
        if val < current_best:
            current_best = val
        best_so_far.append(current_best)
        return val

    init_kwargs = {'budget': budget, 'dim': dim}
    sig = inspect.signature(algo_cls.__init__)
    if 'seed' in sig.parameters:
        init_kwargs['seed'] = seed
    elif 'random_seed' in sig.parameters:
        init_kwargs['random_seed'] = seed

    optimizer = algo_cls(**init_kwargs)
    try:
        optimizer(tracked_func)
    except Exception as e:
        print(f'Run stopped at eval {eval_count} ({e})')

    while len(best_so_far) < budget:
        best_so_far.append(current_best)

    return np.array(best_so_far[:budget], dtype=float)

# 2. Vanilla BOTorch
def run_botorch_vanilla(problem, budget: int, dim: int, seed: int) -> np.ndarray:
    set_all_seeds(seed)

    bounds = np.column_stack([problem.bounds.lb, problem.bounds.ub])
    n_doe = 2 * dim

    opt = Vanilla_BO(budget=budget, n_DoE=n_doe, acquisition_function='expected_improvement', random_seed=seed)
    opt.verbose = False

    opt(problem=problem, dim=dim, bounds=bounds)

    raw_evals = np.array(opt.f_evals, dtype=float)
    best_history = np.minimum.accumulate(raw_evals)

    if len(best_history) < budget:
        last_val = best_history[-1] if len(best_history) > 0 else float('inf')
        best_history = np.pad(best_history, (0, budget - len(best_history)), constant_values=last_val)

    return best_history[:budget]

# 3. CMA-ES
def run_cma_es(problem, budget: int, dim: int, seed: int) -> np.ndarray:
    set_all_seeds(seed)

    lb = np.array(problem.bounds.lb)
    ub = np.array(problem.bounds.ub)

    x0 = np.random.uniform(lb, ub)
    sigma0 = float(np.mean((ub - lb) / 5.0))
    cma_bounds = [lb.tolist(), ub.tolist()]

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': cma_bounds,
        'seed': seed,
        'verbose': -9,
        'maxfevals': budget
    })

    history = []
    best_y = float('inf')

    while not es.stop() and len(history) < budget:
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            if len(history) >= budget:
                break
            y = problem(x)
            fitnesses.append(y)
            if y < best_y:
                best_y = y
            history.append(best_y)
        if len(fitnesses) == len(solutions):
            es.tell(solutions, fitnesses)

    while len(history) < budget:
        history.append(best_y)

    return np.array(history[:budget], dtype=float)

def main():
    cfg, seeds, dim, budget = load_config('configs.yaml')
    out_dir = Path(cfg['logging']['save_dir']) / 'stage2_benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate Stage 1 output
    bbob_scripts = list((Path(cfg['logging']['save_dir']) / 'bbob_f8').glob('best_bbob_f8_*.py'))
    rw_scripts = list((Path(cfg['logging']['save_dir']) / 'real_world_1124').glob('best_real_world_1124_*.py'))
    assert bbob_scripts, 'Missing best_bbob_f8_*.py in outputs/bbob_f8/'
    assert rw_scripts, 'Missing best_real_world_1124_*.py in outputs/real_world_1124/'

    cls_bbob, name_bbob = load_generated_class(bbob_scripts[0])
    cls_rw, name_rw = load_generated_class(rw_scripts[0])

    algorithms = {
        f'Generated_BBOB ({name_bbob})': ('custom', cls_bbob),
        f'Generated_RealWorld ({name_rw})': ('custom', cls_rw),
        'BoTorch_Vanilla': ('botorch', None),
        'CMA_ES': ('cmaes', None),
    }
    target_problems = {
        'BBOB_F8_Rosenbrock': (cfg['targets']['bbob']['function_id'], cfg['targets']['bbob']['instance']),
        'RealWorld_1124_FMSound': (cfg['targets']['real_world']['problem_id'], 1),
    }

    print(f"Stage 2 | Group {cfg['experiment']['group_id']} | dim={dim} | budget={budget} | seeds={seeds[0]}-{seeds[-1]}", flush=True)

    # Build the task list
    tasks = []
    for prob_label, (prob_id, inst_id) in target_problems.items():
        for algo_label, (algo_type, _) in algorithms.items():
            if algo_type == 'custom':
                algo_path = str(bbob_scripts[0] if 'BBOB (' in algo_label else rw_scripts[0])
            else:
                algo_path = None
            for seed in seeds:
                tasks.append((prob_label, algo_label, algo_type, algo_path,
                              prob_id, inst_id, seed, budget, dim))

    n_workers = max(1, min(MAX_WORKERS, (os.cpu_count() or 1) - 1))
    print(f"Running {len(tasks)} tasks with {n_workers} parallel workers\n", flush=True)

    all_records = []
    start_all = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_task, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            prob_label, algo_label = t[0], t[1]
            seed = t[6]
            try:
                _, _, _, curve = fut.result(timeout=RUN_TIMEOUTS[t[2]])
                elapsed = time.time() - start_all
                print(f'  {prob_label} / {algo_label} / seed={seed}  '
                      f'best={curve[-1]:.4e}  (total {elapsed/60:.1f} min)', flush=True)
                for step, val in enumerate(curve):
                    all_records.append({
                        'Problem': prob_label, 'Algorithm': algo_label,
                        'Seed': seed, 'Evaluation': step + 1, 'Best_So_Far': val,
                    })
            except Exception as e:
                print(f'  {prob_label} / {algo_label} / seed={seed}  '
                      f'FAILED: {type(e).__name__}: {e}', flush=True)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(out_dir / 'stage2_convergence_data.csv', index=False)

    summary_df = (
        results_df[results_df['Evaluation'] == budget]
        .groupby(['Problem', 'Algorithm'])['Best_So_Far']
        .agg(mean='mean', std='std', median='median', min='min', max='max')
        .reset_index()
    )
    summary_df.to_csv(out_dir / 'stage2_final_performance.csv', index=False)

    print(f'\nSaved: {out_dir}/stage2_convergence_data.csv')
    print(f'Saved: {out_dir}/stage2_final_performance.csv\n')
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()