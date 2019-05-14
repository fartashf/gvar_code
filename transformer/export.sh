export LD_LIBRARY_PATH=$HOME/.local/lib:/pkgs/cuda-9.0/lib64:/pkgs/cudnn-9.2-v7.3.1/lib64:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export PATH="$HOME/.local/bin:$PATH"
# export PATH=/pkgs/anaconda3/bin:$PATH
export CUDA_HOME=/pkgs/cuda-9.0/  # needed for pytorch
# source activate py_env
# added by Anaconda3 5.3.0 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/scratch/gobi1/fartash/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/pkgs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/pkgs/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/pkgs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<
conda activate py1
