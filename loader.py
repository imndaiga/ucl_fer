if __name__ == '__main__':
	# Residual blocks
	# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
	n = 5

	# Real-time preprocessing of the image data
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Real-time data augmentation
	img_aug = tflearn.ImageAugmentation()
	img_aug.add_random_flip_leftright()

	print('Building Residual Network')
	# Building Residual Network
	net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
	net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
	net = tflearn.residual_block(net, n, 16)
	net = tflearn.residual_block(net, 1, 32, downsample=True)
	net = tflearn.residual_block(net, n-1, 32)
	net = tflearn.residual_block(net, 1, 64, downsample=True)
	net = tflearn.residual_block(net, n-1, 64)
	net = tflearn.batch_normalization(net)
	net = tflearn.activation(net, 'relu')
	net = tflearn.global_avg_pool(net)

	# Regression
	net = tflearn.fully_connected(net, 7, activation='softmax')
	adam = tflearn.optimizers.adam()
	net = tflearn.regression(net, optimizer=adam,
	                         loss='categorical_crossentropy')
	# Training
	model = tflearn.DNN(net, checkpoint_path='models/model_fer',
	                    max_checkpoints=20, tensorboard_verbose=2,
	                    clip_gradients=0.)



	#model.load('drive/model.tfl', weights_only=True)
	# model.load('drive/model.tfl.data-00000-of-00001', weights_only=True)  
	model.load('model.tfl')