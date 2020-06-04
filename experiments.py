from experiments.ga import run as run_ga
from experiments.papernot import run as run_papernot
from experiments.rq2_analysis import run_success_rates
from src.train import run as run_training
import sys, getopt, os


def main(experiments_id):

    if "random" in experiments_id:
        #Random evolution
        config_name, experiment_id = "./configurations/random_fast.json", "randomf4"
        run_ga(config_name, experiment_id)

    if "f1f2f4" in experiments_id:
        #F1F2 evolution
        config_name, experiment_id = "./configurations/config_f1f2_fast.json", "f1f2f4"
        run_ga(config_name, experiment_id)

    if "f1f3f4" in experiments_id:
        #F1F3 evolution
        config_name, experiment_id = "./configurations/config_f1f3_fast.json", "f1f3f4"
        run_ga(config_name, experiment_id)
    
    if "f1f2f3f4" in experiments_id:
        #F1F2F3 evolution
        config_name, experiment_id = "./configurations/config_f1f2f3_fast.json", "f1f2f3f4"
        run_ga(config_name, experiment_id)

    if "papernot" in experiments_id:
        # Iterative papernot attack 
        config_name, experiment_id = "./configurations/random_fast.json", "papernot"
        adv, objectives = run_papernot(config_name, experiment_id)

def init(argv):

    experiment = ["f1f2f3f4","f1f3f4","f1f2f4","random","papernot"]
    try:
        opts, args = getopt.getopt(argv, "hx:", [
                                   "xp="])
    except getopt.GetoptError:
        pass

    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'experiment.py -x <all|f1f2f3f4|f1f3f4|f1f2f4|randomf4|papernot>')
            sys.exit()
        elif opt in ("-x", "--xp"):
            if opt != "all":
                experiment = [opt]

    main(experiment)


if __name__ == "__main__":
    init(sys.argv[1:])

       