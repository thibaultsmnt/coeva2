from experiments.rq2_analysis import run_success_rates


#Random evolution
config_name, experiment_id = "./configurations/random_fast.json", "randomf4"
random_run = run_success_rates(config_name, experiment_id,100)

#F1F2 evolution
config_name, experiment_id = "./configurations/config_f1f2_fast.json", "f1f2f4"
f1f2_run = run_success_rates(config_name, experiment_id,100)

#F1F3 evolution
config_name, experiment_id = "./configurations/config_f1f3_fast.json", "f1f3f4"
f1f3_run = run_success_rates(config_name, experiment_id,100)

#F1F2F3 evolution
config_name, experiment_id = "./configurations/config_f1f2f3_fast.json", "f1f2f3f4"
f1f2f3_run = run_success_rates(config_name, experiment_id,100)

df = random_run.append(f1f2f3_run)
df = df.append(f1f2_run)
df = df.append(f1f3_run)
df.index = df["run_id"]
print(df.transpose().iloc[1:])

#run_training()

