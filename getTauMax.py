
'''Program to calculate tauMax, the collisional lifetime of the largest
bodies at a certain location in a debris disc, based on the model of
Lohne2008. By default the code uses the assumptions in Pearce2023
(submitted) for various relations and parameters, for example the
critical fragmentation energy QD*, but these can be changed.

To use the code, simply set your desired parameters in the 'User Inputs'
section below, then run the code. This will print out tauMax, as well as
some intermediate calculations. Unless you want to change anything more
major like the QD* function, nothing below the User Inputs section needs
changing.

Feel free to use this code, and if the results go into a publication,
then please cite Lohne et al. 2008 and Pearce et al. 2023 (submitted).
Also, please let me know if you find any bugs or have any requests.'''

############################### Libraries ###############################
import sys
import numpy as np
import math
############################## User Inputs ##############################
''' Parameters of the system. Feel free to change these; nothing outside
this section needs changing, unless you want e.g. a different QD*
parameterisation'''

# Star mass, in solar masses
mStar_mSun = 2.

# Radius of the largest debris body, in km
sMax_km = 50.

# Radial location, in au
r_au = 13.22

# Debris eccentricity
e = 0.0433

# Debris inclination, in radians
i_rad = e/2.

# Initial disc surface-density profile: xM*(r/au)^(-alpha) MEarth/au^2.
# For the Minimum-Mass Solar Nebula, xM=1 and alpha=1.5
xM = 1.
alphaR = 1.5

# Size-distribution slope in the gravity regime
qg = 1.67751

# Size-distribution slope in the primordial regime
qp = 1.87

# Debris material density, in grams per cm^3
rho_gramPerCm3 = 3.5

# Smallest dust size, in micron (provided sMin << sMax, this has very
# little effect on tauMax)
sMin_um = 2.

# Number of digits after decimal point to print outputs
nOutputDecimalPlaces = 4

############# Constants (don't change anything below here!) #############
# Gravitational constant * solar mass
GMSun_au3PerYr2 = 4.*np.pi**2

# Unit conversions
km_m = 1e3
km_cm = 1e5
cm_um = 1e5
au_km = 1.496e8
au_cm = au_km * km_cm

mEarth_gram = 5.972e27
yr_s = 3.154e7

GMSun_cm3PerS2 = GMSun_au3PerYr2 * au_cm**3 / yr_s**2

########################### Physics functions ###########################
def GetTauMax_yr(sMax_km, r_au, e, i_rad, mStar_mSun):
	'''Return tauMax (Eq. 42 in Lohne2008). Note that the index -1 around
	the square bracket in that paper is an error, and is omitted here;
	this change has only a minor effect provided sMax >> sMin.'''
	
	# Get f
	f = GetF(e, i_rad)
	
	# Get QDStar
	qDStarForSMax_ergPerGram = GetQDStar_ergPerGram(sMax_km, r_au, f, mStar_mSun)
	
	# Get X_c(s,r)
	xC = GetXc(qDStarForSMax_ergPerGram, r_au, mStar_mSun, f)
	
	# Get G(qg,s,r)
	g = GetG(qg, xC, sMax_km, sMax_km)

	# Get the initial surface-density at this radius
	initialSDAtR_mEarthPerAu2 = GetInitialSDAtR_mEarthPerAu2(r_au, xM, alphaR)

	# Print out the calculated parameters
	PrintCalculatedParameters(f, qDStarForSMax_ergPerGram, xC, g, initialSDAtR_mEarthPerAu2)
		
	# If X_c >= 1 (i.e. G <= 0), tauMax is infinite
	if xC >= 1:
		return np.inf
	
	# Get the initial mass at this radius, divided by the fractional 
	# annulus width dr/r (done this way because dr/r cancels out in the
	# tauMax equation)
	m0DividedByDrOverR_gram = Getm0DividedByDrOverR_gram(initialSDAtR_mEarthPerAu2, r_au)
	
	# Get tauMax in seconds. Note dr/r cancels, because M0 is expressed 
	# as divided by dr/r
	sMax_cm = sMax_km * km_cm
	r_cm = r_au * au_cm
	
	tauMax_s = 16.*np.pi*rho_gramPerCm3 / (3.*m0DividedByDrOverR_gram) * sMax_cm * r_cm**(7./2.) / (GMSun_cm3PerS2*mStar_mSun)**.5 * (qg-5./3.)/(2.-qp) * (1.-(sMin_um/cm_um/sMax_cm)**(6.-3*qp)) * i_rad/(f*g) 
	
	# Convert to years
	tauMax_yr = tauMax_s / yr_s
	
	return tauMax_yr

#------------------------------------------------------------------------
def GetF(e, i_rad):
	'''Get the value f, from debris eccentricity and inclination
	(Eq. 23 in Lohne2008).'''
	
	f = (5./4.*e**2 + i_rad**2)**.5
	
	return f
		
#------------------------------------------------------------------------
def GetQDStar_ergPerGram(s_km, r_au, f, mStar_mSun):
	'''Get the critical fragmentation energy QDStar using the form of
	Krivov2018 (Eq. 1 in that paper).'''
	
	# Constants in equation
	v0_kmPerS = 3.
	As_ergPerGram = 5e6
	Bs_ergPerGram = 5e6
	bs = -0.12
	bg = 0.46
	
	# Calculate collision speed	
	vCol_kmPerS = GetCollisionSpeed_kmPerS(mStar_mSun, r_au, f)
	
	# Get body radius in m
	s_m = s_km * km_m
	
	# Calculate QDStar
	qDStarForSMax_ergPerGram = (vCol_kmPerS/v0_kmPerS)**0.5 * (As_ergPerGram*s_m**(3.*bs) + Bs_ergPerGram*s_km**(3.*bg))

	return qDStarForSMax_ergPerGram
	
#------------------------------------------------------------------------
def GetCollisionSpeed_kmPerS(mStar_mSun, r_au, f):
	'''Get the collision speed.'''

	# Get the collision speed in au per year
	vCol_auPerYr = (GMSun_au3PerYr2*mStar_mSun/r_au)**.5 * f
	
	# Convert to km per sec
	vCol_kmPerS = vCol_auPerYr * au_km / yr_s
	
	return vCol_kmPerS
	
#------------------------------------------------------------------------
def GetXc(qDStarForSMax_ergPerGram, r_au, mStar_mSun, f):
	'''Gets the parameter X_c (Eq. 25 in Lohne2008).'''
		
	# Convert distance to cm
	r_cm = r_au * au_cm

	xC = (2.*qDStarForSMax_ergPerGram * r_cm / (GMSun_cm3PerS2*mStar_mSun * f**2))**(1./3.)
	
	return xC
	
#------------------------------------------------------------------------
def GetG(q, xC, s_km, sMax_km):
	'''Gets the parameter G (Eq. 24 in Lohne2008).'''
	
	g = GetTermInG(q, xC, s_km, sMax_km, 5) + 2.*GetTermInG(q, xC, s_km, sMax_km, 4) + GetTermInG(q, xC, s_km, sMax_km, 3)
	
	return g

#------------------------------------------------------------------------
def GetTermInG(q, xC, s_km, sMax_km, nInTerm):
	'''Gets the one of the three sub-terms used to find the parameter G
	(Eq. 24 in Lohne2008).'''
	
	# Get n-3q
	nMinus3q = float(nInTerm) - 3.*q

	# Get the term in G
	termInG = (5.-3.*q) / nMinus3q * (xC**nMinus3q - (sMax_km/s_km)**nMinus3q)
	
	return termInG

#------------------------------------------------------------------------
def GetInitialSDAtR_mEarthPerAu2(r_au, xM, alphaR):
	'''Get the initial surface density at this radial location. Assumes
	the initial disc has a surface-density profile going as
	~xM * r^-alphaR, where for the Minimum-Mass Solar Nebula, xM=1,
	alphaR=1.5 and SD(1au)=MEarth/au^2.'''

	initialSDAtR_mEarthPerAu2 = xM * r_au**-alphaR

	return initialSDAtR_mEarthPerAu2
	
#------------------------------------------------------------------------
def Getm0DividedByDrOverR_gram(initialSDAtR_mEarthPerAu2, r_au):
	'''Get the initial mass at this radial location, divided by the
	fractional annulus width dr/r (done this way because dr/r cancels out 
	in the tauMax equation).'''

	# Convert surface density to cgs
	initialSDAtR_gramsPerCm2 = initialSDAtR_mEarthPerAu2 * mEarth_gram / au_cm**2
	
	# Convert the surface density to a mass, keeping the dr/r depedence
	# explicit
	r_cm = r_au * au_cm
	
	m0DividedByDrOverR_gram = 2.*np.pi*r_cm**2*initialSDAtR_gramsPerCm2

	return m0DividedByDrOverR_gram
	
########################### Printing functions ##########################
def	PrintInputParameters():
	'''Prints out the program inputs'''

	print('Input parameters:')
	print('     Star mass: %s MSun' % mStar_mSun)
	print('     sMax: %s km' % sMax_km)
	print('     r: %s au' % r_au)
	print('     e: %s' % e)
	print('     i: %s deg' % GetNumberStringInScientificNotation(i_rad*180./np.pi, precision=nOutputDecimalPlaces))
	print('     xM: %s' % xM)
	print('     alphaR: %s' % alphaR)
	print('     qg: %s' % qg)
	print('     qp: %s' % qp)
	print('     rho: %s gram / cm^3' % rho_gramPerCm3)
	print('     sMin: %s um' % sMin_um)
	
	PrintEmptyLine()

#------------------------------------------------------------------------
def	PrintCalculatedParameters(f, QDStarForSMax_ergPerGram, xC, g, initialSDAtR_mEarthPerAu2):
	'''Prints out the parameters calculated in the program'''

	print('Calculated parameters:')
	print('     f: %s' % GetNumberStringInScientificNotation(f, precision=nOutputDecimalPlaces))
	print('     QDStar: %s erg / gram' % GetNumberStringInScientificNotation(QDStarForSMax_ergPerGram, precision=nOutputDecimalPlaces))
	print('     X_c: %s' % GetNumberStringInScientificNotation(xC, precision=nOutputDecimalPlaces))
	print('     G: %s' % GetNumberStringInScientificNotation(g, precision=nOutputDecimalPlaces))
	print('     Initial SD at r: %s MEarth/au^2' % GetNumberStringInScientificNotation(initialSDAtR_mEarthPerAu2, precision=nOutputDecimalPlaces))

	PrintEmptyLine()
	
#------------------------------------------------------------------------
def GetNumberStringInScientificNotation(num, precision=None, exponent=None):
	'''Returns a string representation of the scientific notation of the
	given number formatted, with specified precision (number of decimal 
	digits to show). The exponent to be used can also be specified
	explicitly.	Adapted from
	https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting.'''

	# Catch nans
	if math.isnan(num):	return np.nan
		
	# Default number of decimal digits (in case precision unspecified)
	defaultDecimalDigits = 2

	# Get precision if not defined
	if precision is None:
		precision = defaultDecimalDigits
			
	# Get exoponent if not defined
	if exponent is None:
	
		# Catch case if number is zero
		if num == 0:
			exponent = 0
		
		# Otherwise number is non-zero
		else:
			exponent = int(math.floor(np.log10(abs(num))))
	
	# Get the coefficient
	coefficient = round(num / float(10**exponent), precision)
	
	# Adjust if rounding has taken coefficient to 10
	if coefficient == 10:
		coefficient = 1
		exponent += 1
	
	# Get the output string
	outString = '{0:.{2}f}e{1:d}'.format(coefficient, exponent, precision)

	return outString

#------------------------------------------------------------------------
def PrintEmptyLine():
	'''Print an empty line (done this way to enable compatibility across 
	Python versions)'''
	
	if sys.version_info[0] < 3: print
	else: print()

################################ Program ################################
PrintEmptyLine()

# Print input parameters
PrintInputParameters()

# Get tauMax
tauMax_yr = GetTauMax_yr(sMax_km, r_au, e, i_rad, mStar_mSun)

# Print the result
if math. isinf(tauMax_yr):
	tauMaxString = 'infinite (debris-excitation level too low to destroy largest bodies)'

else:
	tauMaxString = '%s yr' % GetNumberStringInScientificNotation(tauMax_yr, precision=nOutputDecimalPlaces)

print('tauMax: %s' % tauMaxString)
	
PrintEmptyLine()

#########################################################################

