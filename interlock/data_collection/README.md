# Data collection 
Fundamentally, a jigsaw puzzle is just an image cut up into jigsaw pieces. We need our dataset to train approximately the same visual reasoning capabilities that modern computer vision models have. Therefore, it makes sense to start with one of the same huge public datasets that these models are trained on. ImageNet images are, on average, very small and restrict the number of pieces we can cut puzzles into; it makes sense to train the final classifier on a corpus like Meta's SA-1B dataset, which is a large corpus at high resolution. 

Onboarding notebook will be restricted to ~100 piece puzzles, (approximately 10x10) so resolution shouldn't be a huge issue for that; we will load a small subset of ImageNet images. 


