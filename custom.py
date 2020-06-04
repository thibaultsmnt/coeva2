from experiments.ga import run as run_ga
import sys, getopt, os
from src.coeva2.problem_definition import ProblemConstraints

def init(argv):

    config_file  = None
    experiment_id = None
    nb_constraints = None
    path_constraints = None

    try:
        opts, args = getopt.getopt(argv, "hc:i:n:p:", [
                                   "config=", "id=","cnb=", "cpath="])
    except getopt.GetoptError:
        pass
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'custom.py -c <json configuration> [-i <run id> -n <nb constraint> -p <path to custom constraints>]')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_file = arg

        elif opt in ("-i", "--id"):
            experiment_id = arg

        elif opt in ("-n", "--cnb"):
            nb_constraints = arg

        elif opt in ("-p", "--cpath"):
            path_constraints = arg


    if config_file is None:
        print(
            'Please provide a configuration file . At least: "custom.py -c <json configuration>"')
        sys.exit()
        

    if nb_constraints is not None and path_constraints is not None:
        ProblemConstraints.set_custom_constraints(nb_constraints,path_constraints )
    run_ga(config_file, experiment_id)


if __name__ == "__main__":
    init(sys.argv[1:])