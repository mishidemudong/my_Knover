python save_inference_model.py \
    --model  "NSPModel"\
    --task "NextSentencePrediction" \
    --vocab_path "./config/vocab.txt" \
    --init_pretraining_params './nsp_model/step_8500' \
    --spm_model_file "./config/spm.model" \
    --inference_model_path './nsp_model/saved_10_27' \
    --config_path "./config/12L.json"
