# Bayesian Optimization with LLaMEA-BO (Group 11)
This repository contains the  implementation and evaluation for the **Bayesian Optimization 2026–2027 Practical Assignment** (Leiden University). We let a large language model automatically build Bayesian optimization methods. We test them on two hard six dimension black-box problems. We then check whether they still work on a different problem, and compare them with two standard methods.

## 1. Repository Structure

```text
BO/
├── configs.yaml                     # Contain experiment & hyperparameter settings (Group 11)
├── stage1_run.py                    # Stage 1: LLaMEA evolutionary algorithm generation script
├── stage2_benchmark.py              # Stage 2: Generalization evaluation & benchmarking
├── run_analysis_plots.py            # Convergence log extraction & plotting script
├── requirements.txt                 # Exact Python dependency versions
├── .env.example                     # Template for Groq API key configuration
├── baseline/                        # Reference BoTorch baseline
├── llamevol/                        # Evolutionary LLM framework (LLaMEvol, ESPopulation, IOHEvaluator)
├── Practical_Problems/              # Real-world problem suite (Problem 1124)
└── outputs/
    ├── bbob_f8/                     # Stage 1 best algorithm & ESPopulation logs for BBOB F8
    ├── real_world_1124/             # Stage 1 best algorithm & ESPopulation logs for Problem 1124
    ├── stage2_benchmark/            # Stage 2 raw trajectories & summary tables
    └── convergence_analysis/        # Final Stage 1 (AOCC) & Stage 2 (Loss/Objective) figures
```

---

## 2. Environment Setup and Installation

### Prerequisites
* **Python**: 3.11+

### Installation Steps
1. **Create and activate a Python virtual environment**:
   ```powershell
   python -m venv .venv
   ```

2. **Install all required packages**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
---

## 3. Configure the LLM Provider

Stage 1 uses Groq API endpoint to query the free-to-use `openai/gpt-oss-120b` model.

1. **Create a `.env` file** in the root directory from `.env.example`:
   ```powershell
   copy .env.example .env
   ```
2. **Insert your Groq API key** inside `.env`:
   ```env
   GROQ_API_KEY=gsk_your_actual_api_key_here
   ```
3. **Configure model parameters in `configs.yaml`**:
   ```yaml
   stage1_generation:
     llm:
       provider: "groq"
       model_name: "openai/gpt-oss-120b"
       temperature: 0.8
       max_tokens: 4096
       timeout_seconds: 60
   ```

---

## 4. Running LLaMEA

All runs use Group 11 settings (`dimension = 6`, deterministic seeds $s_{11, r} = 10(11) + r \implies [110, 111, \dots, 119]$).

### Stage 1: Evolutionary BO Algorithm Generation
Runs the `(4 + 12)` elitist evolutionary search (`outer_budget = 100` candidate algorithms, `inner_budget = 60` function evaluations per seed across 10 seeds, ranked by AOCC):
```powershell
python stage1_run_2.py
```
* **Outputs**: Saves prompts, LLM responses, `ESPopulation` `.pkl` checkpoints, and the best `.py` classes under `outputs/bbob_f8/` and `outputs/real_world_1124/`.

### Stage 2: Generalization Evaluation
Evaluates all four algorithms (`AdaptiveTrustHybridBO`, `AdaptiveTrustRegionBO`, `BoTorch_Vanilla`, and `CMA_ES`) across both **COCO-BBOB F8 (Rosenbrock, instance 1)** and **Practical Problem 1124 (FM Sound Waves)** for `600` evaluations across seeds `110–119`:
```powershell
python stage2_benchmark.py
```
* **Outputs**: Saves evaluation trajectories (`stage2_convergence_data.csv`) and summary statistics (`stage2_final_performance.csv`) in `outputs/stage2_benchmark/`.

### Convergence Plotting
Extracts Stage 1 AOCC progression logs and generates separate Stage 2 convergence plots:
```powershell
python run_analysis_plots.py
python format_stage1_plot.py
```

---

## 5. Guidance on Configuring Prompts and Hyperparameters

### LLaMEA & Experimental Hyperparameters (`configs.yaml` & `stage1_run_2.py`)

### Experimental Configuration

| Module | Parameter | Value | Source | Guidance |
|:---|:---|:---|:---|:---|
| **Experiment** | `group_id` | `11` | `configs.yaml` | ID of Group 11 |
| **Experiment** | `seeds` | `[110, 111, …, 119]` | `configs.yaml` | Deterministic seeds used in benchmarking |
| **Experiment** | `dimension` | `6` | `configs.yaml` | Dimension of the problems |
| **Target - BBOB** | `suite` | `"BBOB"` | `configs.yaml` | Selects `ioh.get_problem(...)` path |
| **Target - BBOB** | `function_id` | `8` (Rosenbrock) | `configs.yaml` | Choice of BBOB problem |
| **Target - BBOB** | `instance` | `1` | `configs.yaml` | Instance of BOBB problem |
| **Target - RealWorld** | `problem_id` | `1124` (FM Sound Waves) | `configs.yaml` | Choice of Real-word problem |
| **Stage 1 - Outer** | `outer_budget` | `100` | `configs.yaml` | max candidate algorithms |
| **Stage 1 - Inner** | `inner_budget` | `60` | `configs.yaml` | max evals/run, including init |
| **Stage 1 - Metric** | `metric` | `"AOCC"` | `configs.yaml` | For documentation only, evaluator computes AOCC by default |
| **LLM** | `provider` | `groq` | `configs.yaml` | Choice of free tier LLM provider |
| **LLM** | `model_name` | `openai/gpt-oss-120b` | `configs.yaml` | Choice of free tier LLM model |
| **LLM** | `temperature` | `0.8` | `configs.yaml` | Structural diversity across candidates |
| **LLM** | `max_tokens` | `4096` | `configs.yaml` | Code generation length cap |
| **LLM** | `timeout_seconds` | `60` | `configs.yaml` | Per-request API timeout |
| **Stage 2** | `evaluation_budget` | `600` | `configs.yaml` | Max eval per run |
| **Stage 2** | `runs_per_algorithm` | `10` | `configs.yaml` | Number of independent runs |
| **Stage 2** | `baselines` | `botorch_default`, `cma_es` | `configs.yaml` | Benchmark baselines |
| **Logging** | `save_dir` | `./outputs` | `configs.yaml` | Root for Stage 1 and Stage 2 outputs |
| **Population** | `n_parent` | `4` | `stage1_run.py` | Set the number of parents |
| **Population** | `n_offspring` | `12` | `stage1_run.py` | Set the number of offsprings |
| **Population** | `use_elitism` | `True` | `stage1_run.py` | Elitism setting |
| **Prompting** | `is_bo` | `True` | `stage1_run.py`  | Select BO prompt template |
| **Retries** | `n_retry` | `3` | `stage1_run.py` | Number of reprompting per failed candidate |

## 6. Analysis of Generated Algorithms

### 6.1 Target 1 (BBOB F8 Rosenbrock): `AdaptiveTrustHybridBO`

`AdaptiveTrustHybridBO` (`outputs/bbob_f8/best_bbob_f8_AdaptiveTrustHybridBO.py`) is a trust region Expected Improvement optimizer:

1. **Initialization (`_sample`)**: `n_init = max(2·d, 5) = 12` Sobol points (`scipy.stats.qmc.Sobol`, scrambled). After the initial design, sampling switches to Latin Hypercube for all candidate pools.
2. **Surrogate Model (`_fit_gp`)**: sklearn `GaussianProcessRegressor` with `ConstantKernel * Matern(ν=5/2)` + `WhiteKernel`.
3. **Acquisition Function (`_ei`)**: Standard Expected Improvement.
4. **Acquisition Maximization (`_max_ei`)**: pool + L-BFGS-B restart:
   * Sample `n_candidates = max(2000, 100·d) = 2000` LHS points inside the trust region
   * Compute EI on all 2000
   * Take the top 20 by EI as starting points for L-BFGS-B
   * Run 20 L-BFGS-B restarts (each up to 50 iterations) to refine EI
   * Return the point with the highest EI, fallback to the best pool point if all restarts fail
5. **Trust Region (`__call__`)**: Acquisition is confined to a box around `best_x`, with half the width of `tr_radius`. Radius start at = 5.0, updates every iteration: ×1.2 on improvement, ×0.7 on failure (floor 0.1).

### 6.2 Target 2 (Practical Problem 1124): `AdaptiveTrustRegionBO`

`AdaptiveTrustRegionBO` (`outputs/real_world_1124/best_real_world_1124_AdaptiveTrustRegionBO.py`) is a Lower Confidence Bound trust region optimizer:

1. **Initialization (`_initial_design`)**: `init_samples = 2·d = 12` Sobol points (`scipy.stats.qmc.Sobol`, scrambled). Sampling stays on Sobol throughout.
2. **Surrogate Model (`_fit_gp`)**: sklearn `GaussianProcessRegressor` with `ConstantKernel * Matern(ν=5/2)` + `WhiteKernel`.
3. **Acquisition Function (`_ucb`)**: Lower Confidence Bound with custom exploration schedule.
4. **Acquisition Maximization (`__call__`)**: pool + single L-BFGS-B restart:
   * Sample `n_cand = 128` Sobol points inside the trust region
   * Compute UCB on all 128
   * Take the best by UCB as starting point for L-BFGS-B
   * Run 1 L-BFGS-B restart (up to 30 iterations)
5. **Trust Region (`__call__`)**: Acquisition is confined to a box around `centre`, with half width of `radius_frac * (upper - lower)/2 = radius_frac * 5`. `radius_frac` start at 0.5, updates every: ×1.5 on `success_patience=3` consecutive improvement, ×0.5 on `failure_patience=5` consecutive failures (floor 0.05).

## 7. Stage 2 Generalization Summary (600 Evaluations, Seeds 110–119)

| Target Problem | Algorithm | Mean | SD | Median |
| :--- | :--- | :---: | :---: | :---: |
| **BBOB F8 (Rosenbrock)** | `Generated_BBOB (AdaptiveTrustHybridBO)` | **150.57** | **2.54** | **149.38** |
| **BBOB F8 (Rosenbrock)** | `CMA_ES` | 153.35 | 1.00 | 153.54 |
| **BBOB F8 (Rosenbrock)** | `BoTorch_Vanilla` | 197.76 | 16.83 | 197.65 |
| **BBOB F8 (Rosenbrock)** | `Generated_RealWorld (AdaptiveTrustRegionBO)` | 4473.21 | 2498.29 | 3796.07 |
| **Real-World 1124 (FM Sound)** | `CMA_ES` | **1.84** | 0.53 | 1.92 |
| **Real-World 1124 (FM Sound)** | `Generated_RealWorld (AdaptiveTrustRegionBO)` | 2.13 | 0.78 | **1.91** |
| **Real-World 1124 (FM Sound)** | `BoTorch_Vanilla` | 2.19 | **0.40** | 2.27 |
| **Real-World 1124 (FM Sound)** | `Generated_BBOB (AdaptiveTrustHybridBO)` | 2.49 | 0.77 | 2.55 |

## 9. External Code Attribution

* **`llamevol/` & `misc/`**: Adapted from the **LLaMEA / LLaMEA-BO** framework authored by Niki van Stein, Wenhu Li, Thomas Bäck, and Elena Raponi ([https://github.com/BayesOptApp/LLaMEA-BO/](https://github.com/BayesOptApp/LLaMEA-BO/)).
* **`baseline/` & `Practical_Problems/`**: Provided by the Leiden University Bayesian Optimization course instructors (**Iván Olarte Rodríguez**, **Elena Raponi**, and **Santiago García Mayayo**) via [https://github.com/BayesOptApp/BO_Course_Assignment](https://github.com/BayesOptApp/BO_Course_Assignment).