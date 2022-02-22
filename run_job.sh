#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH
#SBATCH --partition main ### specify partition name where to run a job. Any node: ‘main’; NVidia 2080: ‘rtx2080’; 1080: ‘gtx1080’
#SBATCH --time 0-10:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit (7 days)!! Format: D-H:MM:SS
#SBATCH --job-name bp_learning ### name of the job. replace my_job with your desired job name
#SBATCH --output bp_learning-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=tomya@post.bgu.ac.il ### user’s email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=1 ### number of GPUs (can't exceed 8 gpus for now) allocating more than 1 requires the IT team permission
### Print some data to output file ###
echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
### Start your code below ####
module load anaconda ### load anaconda module
source activate bp_learning_env ### activate a conda environment, replace my_env with your conda environment
python ~/repos/BPLearning/train_composite_automata.py ### this command executes jupyter lab – replace with your own command e.g. ‘python my.py my_arg’