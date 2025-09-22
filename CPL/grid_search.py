import argparse
import itertools
import subprocess
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Grid search for CPL full-batch hyperparameters')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions for each parameter set')
    args = parser.parse_args()

    param_grid = {
        '--cpl_lr_c': [2.0, 5.0, 8.0],
        '--cpl_lr_theta': [0.001, 0.0005],
        '--theta_split_ratio': [0.5, 0.7],
        '--cpl_inner_steps': [5, 10],
        '--cpl_outer_steps': [1],
    }

    keys, values = zip(*param_grid.items())
    
    best_coverage = -1
    best_params = None
    target_coverage = 0.95

    results = []

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        
        cmd = ['python', 'CPL/evaluate_cpl_fullbatch.py']
        for key, value in params.items():
            cmd.append(str(key))
            cmd.append(str(value))
        
        cmd.extend([
            '--cpl_n_epochs', '100',
            '--cpl_sigma_annealing', '0.95',
            '--repetitions', str(args.repetitions)
        ])

        print("\n" + "="*50)
        print(f"Running with parameters: {params}")
        print("="*50)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        output = result.stdout
        
        try:
            coverage_line = [line for line in output.split('\n') if line.strip().startswith('Coverage')][0]
            coverage_str = coverage_line.split()[1]
            coverage = float(coverage_str)

            size_line = [line for line in output.split('\n') if line.strip().startswith('Average Set Size')][0]
            avg_size_str = size_line.split()[3]
            avg_size = float(avg_size_str)
            
            results.append({
                'params': params,
                'coverage': coverage,
                'avg_size': avg_size
            })

            print(f"Coverage: {coverage:.4f}, Avg Size: {avg_size:.4f}")

            if abs(coverage - target_coverage) < abs(best_coverage - target_coverage):
                best_coverage = coverage
                best_params = params

        except (IndexError, ValueError) as e:
            print(f"Could not parse output for parameters: {params}")
            print(f"Error: {e}")
            print("STDOUT:")
            print(output)


    print("\n" + "="*50)
    print("Grid Search Complete")
    print("="*50)
    
    results.sort(key=lambda x: abs(x['coverage'] - target_coverage))

    print("Top 5 parameter sets:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Coverage: {result['coverage']:.4f}, Avg Size: {result['avg_size']:.4f}, Params: {result['params']}")

    if best_params:
        print("\nBest parameter set found:")
        print(f"Coverage: {best_coverage:.4f}")
        print(f"Parameters: {best_params}")
    else:
        print("Could not find a best parameter set.")

if __name__ == '__main__':
    main()
