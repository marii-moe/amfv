#!/usr/bin/env bash
#SBATCH --job-name={{ job_name }}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task={{ gpus }}
{% if gpus == 8 %}
#SBATCH --exclusive
{% else %}
#SBATCH --cpus-per-gpu={{ cpus_per_gpu }}
{% endif %}
{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if slurm_resume %}
#SBATCH --requeue
{% endif %}
{% if qos %}
#SBATCH --qos={{ qos }}
{% endif %}
{% if nice is not none %}
#SBATCH --nice={{ nice }}
{% endif %}
{% if mail_type %}
#SBATCH --mail-type={{ mail_type }}
{% endif %}
{% if mail_user %}
#SBATCH --mail-user={{ mail_user }}
{% endif %}
{% if account %}
#SBATCH --account={{ account }}
{% endif %}
#SBATCH --export=ALL
#SBATCH --output="{{ output_dir }}/slurm/job_%j.log"
#SBATCH --error="{{ output_dir }}/slurm/job_%j.log"

set -euo pipefail
mkdir -p "{{ output_dir }}/slurm"

export HF_HOME="{{ hf_home }}"
source "{{ slurm_dir }}/_vllm_start.sh" "{{ model }}" "{{ gpus }}" "{{ vllm_bin }}"

amfv-eval generate \
    --config "{{ config }}" \
    --base-url "${VLLM_BASE_URL}"

touch "{{ marker }}"
