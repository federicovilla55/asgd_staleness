import argparse
import subprocess

def main():
    """
    Main function to run the experiment based on command line arguments.
    It accepts two arguments: test and overparameterization.

    E.g. to run the script:
    `python ASGD/run_tests.py dasgd 200`
    `python -m ASGD.run_tests saasgd 110`

    :param test: The test/experiment to run. Options are 'saasgd', 'dasgd', 'asap_sgd'.
    :param overparam: The overparameterization size. Options are '110', '150', '200'.
    :return: None
    """
    parser = argparse.ArgumentParser(description='Run an experiment with specified test and overparameterization size.')
    parser.add_argument('test', choices=['saasgd', 'dasgd', 'asap_sgd'], help='The test/experiment to run.')
    parser.add_argument('overparam', choices=['110', '150', '200'], help='The overparameterization size.')
    args = parser.parse_args()

    module_path = f'ASGD.experiments.{args.test}'
    cmd = ['python', '-m', module_path, '--overparam', args.overparam]
    
    subprocess.run(cmd)

if __name__ == '__main__':
    main()