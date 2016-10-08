import os
image_batch_size = 1
window_batch_size = 20
pretrained_model = os.path.join('..', 'model-defs', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel')
mean_file = os.path.join('..', 'model-defs', 'pre_trained_models', 'vgg_16layers', 'mean_image.mat')
solver_file = os.path.join('..', 'model-defs', 'roi_pooling_test', 'solver.prototxt')
deploy_file = os.path.join('..', 'model-defs', 'roi_pooling_test', 'deploy.prototxt')


    # solver.net.forward()
    # for i in xrange(window_batch_size):
    #     print np.argmax(label[i, :40])
    #     print np.argmax(label[i, 40:80])
    #     print np.argmax(label[i, 80:])
    #     I = solver.net.blobs['region_pool5_loc'].data
    #     I = I[i, :, :, :].transpose((1,2,0))
    #     I[:, 39, 2] = 1
    #     I[39, :, 2] = 1
    #     cv2.imshow('IMG2', I)
    #     cv2.waitKey(0)