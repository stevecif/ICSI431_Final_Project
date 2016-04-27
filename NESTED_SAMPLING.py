# NESTED SAMPLING MAIN PROGRAM
#
# This is the pythonic version of the original Nested Sampling C code taken from the book "Data Analysis: A Bayesian Tutorial" by D. Sivia and J. Skilling
# Also need to include "apply.c", which is the aplication code, setting int n, int MAX, struct Object, void Prior, void Explore, void Results.

import sys
import math 

UNIFORM	= ((rand()+0.5)/(RAND_MAX+1.0))	# Uniform inside (0,1)
def PLUS(x,y):	return (x>y ? x+math.log(1+math.exp(y-x)) : y+math.log(1+math.exp(x-y))) #logarithmic addition log(exp(x) + exp(y))
# ______________________________________________________________________ 
	#
	# ___________________________________________________________________
	# 
	#def Obj(n): [e] * n	#Collection of n objects
	Obj = []			#Collection of n objects
	Samples = []        #Objects stored for posteriorresults 			
	logwidth = 0.0		#ln(width in prior mass)
	logLstar = 0.0		#ln(Likelihood constraint)
	H = 0.0 			#Information, initially 0
	logZ = -DBL_MAX 	#ln(Evidence Z, initially 0)
	logZnew = 0.0		#Updated logZ
	i = 0				#Object counter
	copy = 0			#Duplicated object
	worst = 0			#Worst object
	nest = 0			#Nested sampling iteration count

	# Set prior objects
	for i in range(n):
		Prior = &Obj[i]
	#Outermost interval of prior mass
	logwidth = math.log(1.0 - math.exp(-1.0 / n))

	#NESTED SAMPLING LOOP___________________________________________________
	for nest in range(MAX):
		#Worst object in collection, with Weight = width * Likelihood
		worst = 0
		for i in range(1, n):
			if Obj[i].logL < Obj[worst].logL:
				worst = i
		Obj[worst].logWt = logwidth + Obj[worst].logL
		#Update Evidence Z and Information H
		logZnew = PLUS(logZ, Obj[worst].logWt)
		H = math.exp(Obj[worst].logWt - logZnew) * Obj[worst].logL + math.exp(logZ - logZnew) * (H + logZ) - logZnew
		logZ = logZnew
		#Posterior Samples(optional)
		Sampes[nest] = Obj[worst]
		#Kill worst object in favour of copy of different survivor
		while True:
		    copy = int((n * UNIFORM) % n) #force 0 <= copy < n
		    if copy == worst and n > 1: #don't kill if n is only 1    
		        break
		logLstar = Obj[worst].logL
		#new likelihood constraint
		Obj[worst] = Obj[copy] #overwrite worst object
		# Evolve copied object within constraint
		Explore(&Obj[worst], logLstar)
		#Shrink interval
		logwidth -= 1.0 / n
			#_______________NESTED SAMPLING LOOP(might be ok to terminate early)
		
	# Exit with evidence Z, information H, and optional posterior Samples
	printf("# iterates = %d\n" % (nest))
	printf("Evidence: ln(Z) = %g +- %g\n" % (logZ, math.sqrt(H / n)))
	printf("Information: H = %g nats = %g bits\n" % (H, H / math.log(2.)))
	Results(Samples, nest, logZ)
	#optional
		return 0