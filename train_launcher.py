"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
import os

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':
	import tensorflow as tf
	os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

	print('--Parsing Config File')
	conf1='./model/ssd_hg_mcam26k/config_mcam_200.cfg'
	pre_model1='./model/dl_tiny/hg_refined_tiny_200'
	pre_model2='./model/dl_mcam_tiny/mcam_mpii_tiny_3low_200'
	conf2='./model/dl_mcam_tiny/config_mcam_200.cfg'
	conf4='./model/floss_new/config_mcam_200.cfg'	
	conf5='./model_newlr/pre_gt_newlr/config.cfg'
	pre_model3='./model_newlr/pred_ssd/pre_ssdbox_100'
	conf6='./model_newlr/pre_ssdbox/config.cfg'
	conf7='./model_newlr/atm_mpii/config.cfg'
	pre_model7='./model_newlr/new_floss_gt/new_floss_gt_100'
	conf8='./model_newlr/pre_boxp/config.cfg'
	conf9='./model_newlr/pre_floss/config.cfg'
	pre_model4='./model_newlr/pre_boxp/pre_boxp0.1_100'
	conf10='./model_newlr/pred_ssdbox/config.cfg'	
	pre_model5='./model_newlr/pre_ssdbox_floss_boxp/pre_ssdbox_boxp0.1_100'
	conf11='./model_newlr/new_minL_gt/config.cfg'
	conf12='./model_newlr/new_floss_gt/config.cfg'
	pre_model6='./model_newlr/new_minL_gt/new_minL_gt_100'

	confw1='./model_webn/held_tiny/config.cfg'
	prew1='./model_webn/tiny_wloss/tiny_wloss_74'
	confw2='./model_webn/held_tiny_wloss/config.cfg'
	prew2='./model_webn/tiny_mcam_/tiny_mcam_3low_78'
	confw3='./model_webn/held_tiny_wloss/config.cfg'
	prew3='./model_webn/8stack/8stack_56'
	confw4='./model_webn/held_untiny/config.cfg'
	confw5='./model_webn/held_tiny_wloss_bn/config.cfg'		

	model_n=None
	conf=confw5
	reset=False
	boxp=0.2
	f_loss=False
	minL=False
	print('boxp= {}\n pre_model= {}\n config= {}\n focal_loss={}\n minL={}\n '.format(boxp,model_n,conf,f_loss,minL))

	params = process_config(conf)
	print('--Creating Dataset')
	dataset = DataGenerator(minL,boxp,params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
	dataset._create_train_table()
	dataset._create_val_table()
	dataset._randomize()
	dataset._create_sets()
	
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	modif=True

	model = HourglassModel(f_loss=f_loss,nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], saver_dir = params['saver_directory'],tiny= params['tiny'], w_loss=params['weighted_loss'] , joints= params['joint_list'],modif=modif)
	model.generate_model()
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'],config=config, dataset = dataset, load=model_n,reset=reset)##load need change
