"""Sample program that runs a dvi algorithm on a sample MDP."""
import numpy as np

from src.differential_value_iteration.algorithms import algorithms
from src.differential_value_iteration.environments import environments
from src.differential_value_iteration.utils import run_alg, draw
from pathlib import Path


def exp_RVI_Evaluation(env, update_rule, alphas, init_v, max_iters, epsilon, ref_idx=0):
	convergence_flags = np.zeros(alphas.__len__())
	for alpha_idx in range(alphas.__len__()):
		alpha = alphas[alpha_idx]
		alg = algorithms.RVI_Evaluation(env, init_v, alpha, ref_idx)
		print(f'{env.name} RVI Evaluation {update_rule} alpha:{alpha}', end=' ')
		convergence = run_alg(alg, update_rule, max_iters, epsilon)
		print(f'Converged? {convergence}')
		convergence_flags[alpha_idx] = convergence
	return convergence_flags


def exp_RVI_Control(env, update_rule, alphas, init_v, max_iters, epsilon, ref_idx=0):
	convergence_flags = np.zeros(alphas.__len__())
	for alpha_idx in range(alphas.__len__()):
		alpha = alphas[alpha_idx]
		alg = algorithms.RVI_Control(env, init_v, alpha, ref_idx)
		print(f'{env.name} RVI Control {update_rule} alpha:{alpha}', end=' ')
		convergence = run_alg(alg, update_rule, max_iters, epsilon)
		print(f'Converged? {convergence}')
		convergence_flags[alpha_idx] = convergence
	return convergence_flags


def exp_DVI_Evaluation(env, update_rule, alphas, betas, init_v, init_r_bar, max_iters, epsilon):
	convergence_flags = np.zeros((alphas.__len__(), betas.__len__()))
	for alpha_idx in range(alphas.__len__()):
		for beta_idx in range(betas.__len__()):
			alpha = alphas[alpha_idx]
			beta = betas[beta_idx]
			alg = algorithms.DVI_Evaluation(env, init_v, init_r_bar, alpha, beta)
			print(f'{env.name} DVI Evaluation {update_rule} alpha:{alpha} beta:{beta}', end=' ')
			convergence = run_alg(alg, update_rule, max_iters, epsilon)
			print(f'Converged? {convergence}')
			convergence_flags[alpha_idx, beta_idx] = convergence
	return convergence_flags


def exp_DVI_Control(env, update_rule, alphas, betas, init_v, init_r_bar, max_iters, epsilon):
	convergence_flags = np.zeros((alphas.__len__(), betas.__len__()))
	for alpha_idx in range(alphas.__len__()):
		for beta_idx in range(betas.__len__()):
			alpha = alphas[alpha_idx]
			beta = betas[beta_idx]
			alg = algorithms.DVI_Control(env, init_v, init_r_bar, alpha, beta)
			print(f'{env.name} DVI Control {update_rule} alpha:{alpha} beta:{beta}', end=' ')
			convergence = run_alg(alg, update_rule, max_iters, epsilon)
			print(f'Converged? {convergence}')
			convergence_flags[alpha_idx, beta_idx] = convergence
	return convergence_flags


def exp_MDVI_Evaluation(env, update_rule, alphas, betas, init_v, init_r_bar, max_iters, epsilon):
	convergence_flags = np.zeros((alphas.__len__(), betas.__len__()))
	for alpha_idx in range(alphas.__len__()):
		for beta_idx in range(betas.__len__()):
			alpha = alphas[alpha_idx]
			beta = betas[beta_idx]
			alg = algorithms.MDVI_Evaluation(env, init_v, init_r_bar, alpha, beta)
			print(f'{env.name} MDVI Evaluation {update_rule} alpha:{alpha} beta:{beta}', end=' ')
			convergence = run_alg(alg, update_rule, max_iters, epsilon)
			print(f'Converged? {convergence}')
			convergence_flags[alpha_idx, beta_idx] = convergence
	return convergence_flags


def exp_MDVI_Control1(env, update_rule, alphas, betas, init_v, init_r_bar, max_iters, epsilon):
	convergence_flags = np.zeros((alphas.__len__(), betas.__len__()))
	for alpha_idx in range(alphas.__len__()):
		for beta_idx in range(betas.__len__()):
			alpha = alphas[alpha_idx]
			beta = betas[beta_idx]
			alg = algorithms.MDVI_Control1(env, init_v, init_r_bar, alpha, beta)
			print(f'{env.name} MDVI Control1 {update_rule} alpha:{alpha} beta:{beta}', end=' ')
			convergence = run_alg(alg, update_rule, max_iters, epsilon)
			print(f'Converged? {convergence}')
			convergence_flags[alpha_idx, beta_idx] = convergence
	return convergence_flags


def exp_MDVI_Control2(env, update_rule, alphas, betas, init_v, init_r_bar, max_iters, epsilon):
	convergence_flags = np.zeros((alphas.__len__(), betas.__len__()))
	for alpha_idx in range(alphas.__len__()):
		for beta_idx in range(betas.__len__()):
			alpha = alphas[alpha_idx]
			beta = betas[beta_idx]
			alg = algorithms.MDVI_Control2(env, init_v, init_r_bar, alpha, beta)
			print(f'{env.name} MDVI Control2 {update_rule} alpha:{alpha} beta:{beta}', end=' ')
			convergence = run_alg(alg, update_rule, max_iters, epsilon)
			print(f'Converged? {convergence}')
			convergence_flags[alpha_idx, beta_idx] = convergence
	return convergence_flags
			
			
if __name__ == '__main__':
	alphas = [1.0, 0.999, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
	betas = [1.0, 0.999, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
	max_iters = 100000
	epsilon = 1e-7
	plots_dir = "plots/"
	Path(plots_dir).mkdir(parents=True, exist_ok=True)
	envs = [environments.mrp1, environments.mrp2, environments.mrp3]
	for env in envs:
		init_v = np.zeros(env.num_states())
		init_r_bar_scalar = 0
		init_r_bar_vec = np.zeros(env.num_states())
		results = exp_RVI_Evaluation(env, 'exec_sync', alphas, init_v, max_iters, epsilon, ref_idx=0)
		draw(results, plots_dir + env.name + '_RVI_Evaluation_sync', alphas)
		results = exp_RVI_Evaluation(env, 'exec_async', alphas, init_v, max_iters, epsilon, ref_idx=0)
		draw(results, plots_dir + env.name + '_RVI_Evaluation_async', alphas)
		results = exp_DVI_Evaluation(env, 'exec_sync', alphas, betas, init_v, init_r_bar_scalar, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_DVI_Evaluation_sync', alphas, betas)
		results = exp_DVI_Evaluation(env, 'exec_async', alphas, betas, init_v, init_r_bar_scalar, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_DVI_Evaluation_async', alphas, betas)
		results = exp_MDVI_Evaluation(env, 'exec_sync', alphas, betas, init_v, init_r_bar_vec, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_MDVI_Evaluation_sync', alphas, betas)
		results = exp_MDVI_Evaluation(env, 'exec_async', alphas, betas, init_v, init_r_bar_vec, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_MDVI_Evaluation_async', alphas, betas)
	
	envs = [environments.mdp2]
	for env in envs:
		init_v = np.zeros(env.num_states())
		init_r_bar_scalar = 0
		init_r_bar_vec = np.zeros(env.num_states())
		results = exp_RVI_Control(env, 'exec_sync', alphas, init_v, max_iters, epsilon, ref_idx=0)
		draw(results, plots_dir + env.name + '_RVI_Control_sync', alphas)
		results = exp_RVI_Control(env, 'exec_async', alphas, init_v, max_iters, epsilon, ref_idx=0)
		draw(results, plots_dir + env.name + '_RVI_Control_async', alphas)
		results = exp_DVI_Control(env, 'exec_sync', alphas, betas, init_v, init_r_bar_scalar, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_DVI_Control_sync', alphas, betas)
		results = exp_DVI_Control(env, 'exec_async', alphas, betas, init_v, init_r_bar_scalar, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_DVI_Control_async', alphas, betas)
		results = exp_MDVI_Control1(env, 'exec_sync', alphas, betas, init_v, init_r_bar_vec, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_MDVI_Control1_sync', alphas, betas)
		results = exp_MDVI_Control1(env, 'exec_async', alphas, betas, init_v, init_r_bar_vec, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_MDVI_Control1_async', alphas, betas)
		results = exp_MDVI_Control2(env, 'exec_sync', alphas, betas, init_v, init_r_bar_vec, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_MDVI_Control2_sync', alphas, betas)
		results = exp_MDVI_Control2(env, 'exec_async', alphas, betas, init_v, init_r_bar_vec, max_iters, epsilon)
		draw(results, plots_dir + env.name + '_MDVI_Control2_async', alphas, betas)
