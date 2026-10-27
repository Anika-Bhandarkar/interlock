# Data collection 
Fundamentally, a jigsaw puzzle is just an image cut up into jigsaw pieces. We need our dataset to train approximately the same visual reasoning capabilities that modern computer vision models have. Therefore, it makes sense to start with one of the same huge public datasets that these models are trained on. ImageNet images are, on average, very small and restrict the number of pieces we can cut puzzles into; it makes sense to train the final classifier on a corpus like Meta's SA-1B dataset, which is a large corpus at high resolution. 

We started with Imagenette (https://github.com/fastai/imagenette), a small ImageNet subset, and have since switched to SA-1B. Onboarding notebook is restricted to ~100 piece puzzles (approximately 10x10) cut from 400x400 crops, so only a small subset of SA-1B is downloaded (1,800 images by default; see `scripts/download.py`).


