import os
import re
from pathlib import Path
import yaml
from dotenv import load_dotenv
load_dotenv()
import logging
#import torch
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.prompt_generators import BaselinePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger
from Practical_Problems.Practical_Problems_Suite import get_practical_problem
#torch.set_num_threads(1)
def run_stage1_for_target(target_name: str, problem_id: int, instance_id: int, problem_desc: str, cfg: dict):
    exp_cfg = cfg['experiment']
    gen_cfg = cfg['stage1_generation']
    output_dir = Path(cfg['logging']['save_dir']) / target_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Stage 1: {target_name} | problem {problem_id} ===')

    # 1. Instantiate Evaluator
    evaluator = IOHEvaluator(budget=gen_cfg['inner_budget'], dim=exp_cfg['dimension'], problems=[problem_id], instances=[[instance_id]], repeat=len(exp_cfg['seeds']), seeds=exp_cfg['seeds'])
    #evaluator.max_eval_workers = 10
    evaluator.timeout = 30 * 60

    # 2. Instantiate Prompt Generator
    prompt_generator = BaselinePromptGenerator()
    prompt_generator.is_bo = True
    prompt_generator.problem_desc = problem_desc

    # 3. Instantiate LLM Manager
    model_name = gen_cfg['llm']['model_name']
    llm_manager = LLMmanager(model_key=model_name)

    # 4. Instantiate Population
    es_options = {
        'n_parent': 4,
        'n_offspring': 12,
        'is_elitist': True,
        'log_dir': str(output_dir)
    }
    population = ESPopulation(
        n_parent=es_options['n_parent'],
        n_offspring=es_options['n_offspring'],
        use_elitism=es_options['is_elitist']
    )
    population.save_dir = es_options['log_dir']
    population.name = f"evol_{target_name}_dim{exp_cfg['dimension']}"
    population.debug_save_on_the_fly = True

    # 5. Instantiate LLaMEvol orchestrator
    llamevol = LLaMEvol()

    # 6. Run the evolution
    llm_params = {
        'temperature': gen_cfg['llm']['temperature'],
        'max_tokens': gen_cfg['llm']['max_tokens']
    }
    llamevol.run_evolutions(
        llm=llm_manager,
        evaluator=evaluator,
        prompt_generator=prompt_generator,
        population=population,
        n_population=gen_cfg['outer_budget'],
        n_retry=3,
        options={'llm_params': llm_params}
    )

    # 7. Save the final population and best algorithm
    population.save(suffix='_final')

    best_individual = population.get_best_of_all(maximize=True)
    if best_individual is not None:
        handler = population.get_handler_from_individual(best_individual)
        safe_name = re.sub(r'[^\w\-.]', '_', handler.code_name or 'unnamed')
        best_code_path = output_dir / f'best_{target_name}_{safe_name}.py'
        with open(best_code_path, 'w', encoding='utf-8') as f:
            f.write(handler.code)
        print(f'Best algorithm saved to: {best_code_path}')
        print(f'Best Algorithm: {handler.code_name} (AOCC: {best_individual.fitness:.4f})')

    print(f'Evolution finished. Results saved in: {population.save_dir}')

def main():
    # Configure logging
    setup_logger(level=logging.INFO, color=True)

    # Load configuration
    with open('configs.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # API key check
    if not os.getenv('GROQ_API_KEY'):
        raise EnvironmentError('GROQ_API_KEY is missing.')

    # # Target 1: BBOB function
    bbob_cfg = cfg['targets']['bbob']
    run_stage1_for_target(
        target_name='bbob_f8',
        problem_id=bbob_cfg['function_id'],
        instance_id=bbob_cfg['instance'],
        problem_desc='BBOB Function 8 (Rosenbrock, 6D)',
        cfg=cfg
    )

    # Target 2: Practical Problem 1124 (FM Sound Waves)
    rw_cfg = cfg['targets']['real_world']
    run_stage1_for_target(
        target_name='real_world_1124',
        problem_id=rw_cfg['problem_id'],
        instance_id=1,
        problem_desc='Real World Problem 1124 (Frequency Modulated Sound Waves, 6D)',
        cfg=cfg
    )

if __name__ == '__main__':
    main()