#BSUB -P acc_bedsore_images
#BSUB -q gpu
#BSUB -gpu num=1
#BSUB -R a100
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -W 12:00
#BSUB -o datathon_train_immune.out
#BSUB -e datathon_train_immune.err

set -euo pipefail

# Modules
ml purge
ml cuda/11.8.0
ml anaconda3/2024.06

# Conda activate (non-interactive safe)
eval "$(conda shell.bash hook)"
conda activate pressureulcerstudy

# Ensure CUDA runtime is visible for TF
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Thread/env knobs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

# Train
python -u train_and_eval_immune.py