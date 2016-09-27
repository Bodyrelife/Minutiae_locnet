import os
image_batch_size = 2
window_batch_size = 128
pretrained_model = os.path.join('..', 'model-defs', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel')
mean_file = os.path.join('..', 'model-defs', 'pre_trained_models', 'vgg_16layers', 'mean_image.mat')
solver_file = os.path.join('..', 'model-defs', 'VGG16_LocNet_40_40_360', 'solver.prototxt')
deploy_file = os.path.join('..', 'model-defs', 'VGG16_LocNet_40_40_360', 'deploy.prototxt')