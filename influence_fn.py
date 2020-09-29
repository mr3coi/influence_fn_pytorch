import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
import numpy as np
from time import time

from influence_fn_pytorch.inverse_hvp import get_inverse_hvp

def get_influence(
		model,
		X_train, X_test,
		Y_train=None, Y_test=None,
		criterion=nn.MSELoss(),
		preproc_data_fn=None,
		approx_type='lissa',
		approx_params=None,
		verbose=False):
	params = list(model.parameters())

	if verbose:
		start_time = time()

	# 1. Compute gradient at each training point
	train_grads = []
	for idx, (x_train, y_train) in enumerate(zip(X_train, Y_train)):
		if preproc_data_fn is not None:
			x_train, y_train = preproc_data_fn(x_train, y_train)
		loss = criterion(model(x_train), y_train)
		grads = grad(loss, params)
		train_grads.append(grads)

	train_complete = time()
	if verbose:
		print(f">>> Computing training gradients complete, "
				 f"duration: {train_complete - start_time:.2f} seconds")

	IF_values = []
	for idx, (x_test, y_test) in enumerate(zip(X_test, Y_test)):
		# 2. Compute `s_test` for each test point
		if preproc_data_fn is not None:
			x_test, y_test = preproc_data_fn(x_test, y_test)
		loss = criterion(model(x_test), y_test)
		test_grads = grad(loss, params)

		s_test = get_inverse_hvp(model, criterion, list(zip(X_train,Y_train)), test_grads,
											approx_type=approx_type,
											approx_params=approx_params,
											preproc_data_fn=preproc_data_fn)
		for item in s_test:
			assert not torch.isnan(item).any(), \
				f">>> ERROR: `s_test` contains nan (test idx: {idx+1})"

		# 3. Combine #1 and #2 to obtain influence functions
		inf_up_loss = []
		for train_grad in train_grads:
			inf = 0
			for train_grad_p, s_test_p in zip(train_grad, s_test):
				assert train_grad_p.shape == s_test_p.shape
				inf += -torch.sum(train_grad_p * s_test_p)	# Parameter-wise dot product accumulation
			inf_up_loss.append(inf)

		IF_values.append(inf_up_loss)
		
		if verbose and (idx+1) % 10 == 0:
			print(f">>> Computed {idx+1}th test point")

	if verbose:
		print(f">>> Computing IFs complete, "
				 f"duration: {time() - train_complete:.2f} seconds")
		
	return torch.tensor(IF_values)


def get_influence2(
		model,
		train_dataset,
		test_dataset,			# Should account for `test_indices`
		train_indices=None,		# `train_idx`
		#test_indices=None,		# `test_indices` => unnecessary
		criterion=nn.MSELoss(),
		test_fn=nn.MSELoss(),
		is_scalar=False,
		batch_size=None,
		train_batch_size=1,
		test_batch_size=1,
		collate_fn=None,
		approx_type='lissa',
		approx_params=None,
		train_has_label=True,
		test_has_label=True,
		verbose=False):
	'''Computes the $inf_{up,loss}$ of each training point specified
	in `train_indices` with respect to all test points in `test_dataset`.
	Close translation of `get_influence_on_test_loss` method in the
	original source code.

	NOTE: Set `criterion` to `None` to calculate IF w.r.t. model output
		  (which must be a scalar)

	:train_dataset: Contains train data points used to train the model.
			Of type `torch.utils.data.Dataset` or its subclass.
	:test_dataset: Contains test data points of interest
	 		(corresponds to `test_indices` in original code).
			Of type `torch.utils.data.Dataset` or its subclass.
	:train_indices: Indices of training points for which influence is computed.
			Set to all possible indices in `train_dataset` if unspecified.
	'''
	params = [param for param in model.parameters() if param.requires_grad]

	if batch_size is not None:
		train_batch_size = test_batch_size = batch_size

	if test_fn is None:
		assert is_scalar, "Model must output a scalar value"
		assert not test_has_label, "Test label requires test_fn"

	test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
							 shuffle=False, collate_fn=collate_fn)

	# 1. Compute test gradients
	test_grads = None	# `test_grad_loss_no_reg_val`
	batch_loss = None	# `temp`
	test_grad_start = time()
	test_grads = None

	for test_batch in test_loader:
		if test_fn is not None:
			if test_has_label:
				batch_x, batch_y = test_batch
				batch_out = model(batch_x)
				assert batch_out.shape == batch_y.shape, \
					"Model output and label should have the same shape"
				batch_out = test_fn(batch_out, batch_y)
			else:
				batch_x = test_batch[0]
				batch_out = test_fn(model(batch_x))
		else:
			batch_x = test_batch[0]
			batch_out = torch.mean(model(batch_x))
		assert batch_out.shape == torch.Size([]), "Agg. test output should be a scalar"

		batch_grads = grad(batch_out, params)
		if test_grads is None:
			test_grads = [g * batch_x.shape[0] for g in batch_grads]
		else:
			test_grads = [g0 + g1 * batch_x.shape[0]
						  for g0,g1 in zip(test_grads, batch_grads)]

	test_grads = [g / len(test_dataset) for g in test_grads]

	if verbose:
		print(">>> Computing test gradients complete, "
			  f"duration: {time() - test_grad_start:.2f} seconds")

	# 2. Compute IHVPs using step 1
	ihvp_start = time()

	# TODO: Implement saving & loading
	inverse_hvp = get_inverse_hvp(
					model, criterion, train_dataset,
					vs=test_grads,
					approx_type=approx_type,
					approx_params=approx_params,
					collate_fn=collate_fn,
					has_label=train_has_label)

	if verbose:
		print(">>> Computing inverse HVPs complete, "
			  f"duration: {time() - ihvp_start:.2f} seconds")

	# 3. Compute inf_up_loss for individual training points
	IF_values = []
	inf_start = time()

	# Consider each of the whole training set if not specified
	if train_indices is not None:
		train_dataset = train_dataset[train_indices]
	train_loader = DataLoader(train_dataset,
							  batch_size=train_batch_size,
							  shuffle=False,
							  collate_fn=collate_fn)

	for idx, train_pt in enumerate(train_loader):
		#train_pt = train_dataset[train_idx:(train_idx+1)]	# To keep batch dimension
		single_loss = None
		if criterion is not None:
			if train_has_label:
				input, target = train_pt
				single_loss = criterion(model(input), target)
			else:
				input = train_pt[0]
				single_loss = criterion(model(input))
		else:
			input = train_pt[0]
			single_loss = torch.mean(model(input).view(-1))
		single_grad = grad(single_loss, params)
		IF_values.append(torch.sum(
					torch.stack(
						[torch.sum(ihvp_p * s_grad_p)
						for ihvp_p, s_grad_p in zip(inverse_hvp, single_grad)]
						)
					))

	if verbose:
		print(">>> Completed computing inf_up_losses, "
			  f"duration: {time() - inf_start:.2f} seconds")

	return torch.tensor(IF_values)
