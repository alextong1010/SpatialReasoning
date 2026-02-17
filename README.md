# SpatialReasoning

BUILDING ENV:
conda create -n spatial python=3.10
conda activate spatial
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install datasets
pip install tqdm

and then pip install whatever else is missing 
