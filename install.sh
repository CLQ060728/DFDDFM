mkdir -p pre_trained/OPENAI_CLIP/
mkdir -p pre_trained/APPLE_DFN_CLIP/
mkdir -p pre_trained/META_DINOV2/
mkdir -p pre_trained/META_DINOV3/LVD/
mkdir -p pre_trained/META_DINOV3/SAT/

pip install --upgrade -r requirements.txt

gdown "https://drive.google.com/drive/folders/1fm3Jd8lFMiSP1qgdmsxfqlJZGpr_bXsx?usp=drive_link" -O pre_trained/OPENAI_CLIP/ --folder
huggingface-cli download "apple/DFN2B-CLIP-ViT-L-14-39B" --local-dir pre_trained/APPLE_DFN_CLIP/ --cache-dir pre_trained/APPLE_DFN_CLIP/temp/
huggingface-cli download "facebook/dinov2-large" --local-dir pre_trained/META_DINOV2/ --cache-dir pre_trained/META_DINOV2/temp/
huggingface-cli download "facebook/dinov3-vitl16-pretrain-lvd1689m" --local-dir pre_trained/META_DINOV3/LVD/ --cache-dir pre_trained/META_DINOV3/LVD/temp/
huggingface-cli download "facebook/dinov3-vitl16-pretrain-sat493m" --local-dir pre_trained/META_DINOV3/SAT/ --cache-dir pre_trained/META_DINOV3/SAT/temp/

rm -rf pre_trained/APPLE_DFN_CLIP/temp/
rm -rf pre_trained/META_DINOV2/temp/
rm -rf pre_trained/META_DINOV3/LVD/temp/
rm -rf pre_trained/META_DINOV3/SAT/temp/
pip cache purge