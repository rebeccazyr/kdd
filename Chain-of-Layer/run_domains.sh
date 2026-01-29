cat > run_domains.sh <<'EOF'
#!/usr/bin/env bash

for domain in ccs google food; do
  echo "=== $domain ==="
  python infer.py \
    --taxo_name "$domain" \
    --model gpt-5 \
    --numofExamples 5 \
    --run True \
    --ChainofLayers True \
    --iteratively True \
    --filter_mode None \
    --save_path_model_response "../results/taxo_${domain}/" \
    --demo_path "./demos/demo_wordnet_train/"
done
EOF