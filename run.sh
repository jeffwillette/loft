DATA_DIR=/d1/dataset/loft
BASE_DIR=/c2/jeff/cache_search/third_party/loft
DATASET=hotpotqa
MODEL=llama3.2-1b-instruct
LEN=128k

python run_inference.py \
    --prompt_prefix_path ${DATA_DIR}/prompts/retrieval_128k/retrieval_${DATASET}_128k.txt \
    --data_dir ${DATA_DIR}/data/retrieval/${DATASET}/${LEN} \
    --model_url_or_name ${MODEL} \
    --split dev \
    --context_length ${LEN} \
    --output_path ${BASE_DIR}/outputs/retrieval/${DATASET}/${MODEL}/${LEN}/predictions.jsonl \

python run_evaluation.py \
    --answer_file_path ${DATA_DIR}/data/retrieval/${DATASET}/${LEN}/dev_queries.jsonl \
    --pred_file_path ${BASE_DIR}/outputs/retrieval/${DATASET}/${MODEL}/${LEN}/predictions.jsonl \
    --task_type retrieval
