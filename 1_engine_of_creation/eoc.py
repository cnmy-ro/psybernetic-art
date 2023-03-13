"""
Engine of Creation
"""

__author__ = "Chinmay Rao"


import random
import numpy as np
from utils import *


random.seed(0)
np.random.seed(0)



def run(contours, masks, art, automata_patterns, portraits, spectra, signals, writer):
	"""
	Run the engine.
	"""

	# 1. Chaos
	print("Computing scene 1 ...")
	frame = scene_1(writer)

	# 2. Fluctuations and first automata
	print("Computing scene 2 ...")
	frame = scene_2(frame, writer)

	# 3. Paradigm shift
	print("Computing scene 3 ...")
	frame = scene_3(frame, spectra, signals, writer)

	# 4. Habitat
	print("Computing scene 4 ...")
	frame = scene_4(frame, masks, contours, writer)

	# 5. Harmonic Synthesis
	print("Computing scene 5 ...")
	frame = scene_5(frame, spectra, signals, writer)

	# 6. Carteian refinement
	print("Computing scene 6 ...")
	frame = scene_6(frame, contours, writer)

	# 7. Homeostasis
	print("Computing scene 7 ...")
	frame, automata = scene_7(frame, contours, masks, art, automata_patterns, writer)

	# 8. Abstraction
	print("Computing scene 8 ...")
	scene_8(frame, automata, masks, portraits, writer)

	



def main():

	contours = load_contours()
	masks = load_masks()
	art = load_art()
	automata_patterns = load_automata_patterns()
	portraits = load_portraits()
	
	signals = encode_contours_into_signals(contours)
	spectra = compute_dtfs_of_signals(signals)

	writer = FramesWriter()
	run(contours, masks, art, automata_patterns, portraits, spectra, signals, writer)



if __name__ == '__main__':
	main()