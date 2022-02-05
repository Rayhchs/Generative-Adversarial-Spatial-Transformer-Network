"""
=============================================================================

							Parameter Guideline
		Input size: Input image size.
		Save epoch: Save weight per save_epoch.
		Save path: Direction of testing output.

=============================================================================
"""


params = {
	'input_size': 64, # Input size
    'batch_size': 1, # Batch size.
    'epochs': 5000, # Number of epochs to train for.
    'learning_rate': 0.00005, # Learning rate.
    'beta1': 0.9, # beta1 of Adam optimizer
    'beta2': 0.999, # beta2 of Adam optimizer
    'save_epoch' : 100, # Save checkpoint per save_epoch
    'save_path': 'C:\\Users\\User\\Desktop\\STNGAN\\result'} # Dir to save testing results