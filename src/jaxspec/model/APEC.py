from xspec import *
from astropy.io import fits
from astropy import units as u
import numpy as np
import time
#import haiku as hk
#import jax.numpy as jnp
#from haiku.initializers import Constant
import matplotlib.pyplot as plt



class APEC():
    
    def __init__(self):
        """
        Initialize the continuum and line files loaded for the APEC interpolation.
        Also loads the abundance files.

        Returns
        -------
        None.

        """
        self.apec = fits.open('/Users/xifuxcat/Documents/heasoft-6.31/spectral/modelData/apec_v3.0.9_coco.fits')
        self.apeclines = fits.open('/Users/xifuxcat/Documents/heasoft-6.31/spectral/modelData/apec_v3.0.9_line.fits')
        self.params = self.apec[1].data
        self.paramslines = self.apeclines[1].data
        self.trace_elements = np.array([3,4,5,9,11,15,17,19,21,22,23,24,25,27,29,30])-1
        self.metals = np.array([6, 7, 8, 10, 12, 13, 14, 16, 18,20, 26, 28])-1

    def __call__(self, energybins, T, Z, norm, addpseudo = False, 
                 addlines = False):
        """
        General function to provide the APEC spectrum. Options are to include
        the pseudo continuum and/or the lines.
        

        Parameters
        ----------
        energybins : ARRAY
            Edges of the bins at which the spectrum is provided.
        T : SCALAR
            Temperature of the model.
        Z : SCALAR
            Metal abundance. Elements included are C, N, O, Ne, Mg, Al, Si, 
            S, Ar, Ca, Fe, Ni. 
        norm : SCALAR
            Norm of the model.
        addpseudo : BOOL, optional
            Whether to add the pseudo continuum. The default is False.
        addlines : BOOL, optional
            Whether to add lines. The default is False.

        Returns
        -------
        spectrum : ARRAY
            APEC spectrum in photons/cm3/s per bin in each bin.

        """
        
        ### Save parameters
        self.Ebins = energybins
        self.T = T
        self.Z = Z
        self.norm = norm
        self.addpseudo = addpseudo
        
        ### Get index of temperature
        idx = np.where(self.params['kT'] > T)[0][0] 
    
        ### Getting the continuum at the data temperatures for interpolation.
        ### Temperature indexes are offset by 2 because of the file structure.
        cont_T1 = self.BinInterp(idx+2-1)
        cont_T2 = self.BinInterp(idx+2)

        ### Temperatures of the extracted continua.
        T1 = self.params['kT'][idx-1] 
        T2 = self.params['kT'][idx]

        ### Temperature interpolation
        spectrum = cont_T1 + (T-T1)*(cont_T2 - cont_T1)/(T2 -T1)
        
        if addlines: 
            idx = np.where(self.paramslines['kT'] > T)[0][0] 
            #print(idx)
            
            T1 = self.paramslines['kT'][idx-1] 
            T2 = self.paramslines['kT'][idx]
            
            lines1 = self.binslines(idx-1 +2)
            lines2 = self.binslines(idx +2)
            
            spectrum += lines1 + (T-T1)*(lines2 - lines1)/(T2-T1) 
            
            #plt.semilogx(Elist, lines1-lines2)
        
        return spectrum
    
    
    def BinInterp(self, idx):
        """
        Using interpolation and integration, this function provides the continuum
        flux in each bin at the given index in the continuum file.

        Parameters
        ----------
        idx : INT
            Index in the apec continuum file at which the continuum should be 
            extracted over the energy bins.

        Returns
        -------
        ARRAY
            Continuum and pseudo continuum flux in photons/cm3/s in each bin.

        """

        ### Correcting by abundance
        corr_abund = np.ones(30) 
        #Only the metals are affected by the abundance
        corr_abund[self.metals] *= self.Z
        
        ### Load continuum
        Ncont = self.apec[idx+2].data['N_Cont'] #Continuum range
        E1 = self.apec[idx].data['E_Cont']      #Continuum energies
        cont = self.apec[idx].data['Continuum'] #Continuum values

        Npseudo = self.apec[idx].data['N_Pseudo'] #P-continuum range
        Epseudo = self.apec[idx].data['E_Pseudo'] #P-continuum energies   
        pseudo = self.apec[idx].data['Pseudo']    #P-continuum values

        ### Interating over elements
        tot_cont = np.zeros(len(self.Ebins)-1)
        tot_pseudo = np.zeros(len(self.Ebins)-1)
        
        for k in range(30):

            # Add continuum for element k and multiply by abundance of k
            tot_cont += self.interpbis(E1[k][:Ncont[k]], 
                                         cont[k][:Ncont[k]])*corr_abund[k] 
                                         
            
            # Add pseudo-continuum for element k and multiply by abundance of k
            if self.addpseudo :
                tot_pseudo += self.interpbis(Epseudo[k][:Npseudo[k]], 
                                               pseudo[k][:Npseudo[k]])*corr_abund[k] 
                                               
        
        #Return sum of contributions weighted by the norm and the 1e14 convention.
        return (tot_cont + tot_pseudo)*1e14*self.norm
    
    def interpbis(self, energies, continuum):
        """
        Interpolates and integrates the continuum provided in the data at the 
        energy bins provided by the user.
        This accounts for the integral of the flux between bin edges and 
        data values which my not be accounted for by simple interpolation.

        Parameters
        ----------
        energies : ARRAY
            Energies of the data values.
        continuum : ARRAY
            Continuum or pseudo-continuum at the energies.


        Returns
        -------
        ARRAY
            Value of the integrated flux in each bin, where the continuum at 
            each bin edge has been interpolated linearly between the data points
            and the integral is accounting for the presence of data points 
            between the bin edges.

        """
    
        N = len(self.Ebins)
    
        ### If the continuum has 0 length, this element has no continuum at this
        ### temperature over this range. Return array of zeros.
        if len(continuum) == 0 :
            return np.zeros(N-1)
        
        ### Mask the data points outside of the energy bins
        mask = np.where((energies>self.Ebins[0])&(energies<self.Ebins[-1]))
        
        ### Identify where the data points should be inserted in the input bins
        sortidx = np.searchsorted(self.Ebins, energies[mask])
        
        ### Interpolate the continuum at the bin edges.
        continterp = np.interp(self.Ebins, energies, continuum)

        ### Insert data points in the corresponding spots in the energies and 
        ### the continuum.
        fullcont = np.insert(continterp, sortidx, continuum[mask])
        fulle = np.insert(self.Ebins, sortidx, energies[mask])        

        ### Integrate the list with the inserted points.
        integ = (fullcont[1:] + fullcont[:-1])/2*(fulle[1:]-fulle[:-1])
        
        ### Integrate the list without the inserted points.
        integbis = (continterp[1:] + continterp[:-1])/2*(self.Ebins[1:]-self.Ebins[:-1])

        ### Identify the indexes of the inserted points in the full list.
        sort_tot1 = np.searchsorted(fulle, energies[mask])

        ### Total output list
        tot = integbis 

        ### The spots at which the data points have been inserted are corrected 
        ### with the appropriate flux. 
        ### This accounts for the area of the curve  between bin edges and data 
        ### points.
        tot[sortidx-1] = integ[sort_tot1-1] + integ[sort_tot1]

        return tot
    
    def binslines(self,idx):
        """
        Returns the flux of the lines in the energy bins. Lines are assumed to
        be infinitely thin in the current version. Their flux is therefore 
        simply added in each corresponding bin.

        Parameters
        ----------
        idx : INT
            Index in the apec lines file at which the lines should be 
            extracted over the energy bins.

        Returns
        -------
        ARRAY
            Flux of the lines in the energy bins provided by the user, in 
            photons/cm3/s

        """
    
        ### Load lines and their wavelengths
        Lines1 = self.apeclines[idx].data['Lambda']*u.Angstrom
        ### Wavelengths in Angstroms are converted to keV
        E1 = Lines1.to(u.keV , equivalencies=u.spectral()).value
        Emiss1 = self.apeclines[idx].data['Epsilon']
        Elements1 = self.apeclines[idx].data['Element']
        
        ### Get the indexes of the lines of the differente elements.
        Elements_idx = [np.where(Elements1 == k) for k in range(1,31)]
        
        ### Correct by abundance
        corr_abund = np.ones(30)
        #Only the metals are affected by the abundance
        corr_abund[self.metals] *= self.Z
        
        ### The emissions are weighted by the abundance.
        Emiss1_weight = np.hstack([Emiss1[Elements_idx[k]]*corr_abund[k] for k in range(30)])
        
        ### Only account for the lines within the energy range of the bins.
        mask = np.where((E1 > self.Ebins[0]) & (E1 < self.Ebins[-1]))
        e = E1[mask]
        emiss = Emiss1_weight[mask]
        
        ### Where to insert each line
        insert_idx = [np.where((self.Ebins > e[k]))[0][0]-1 for k in range(len(e))]
        

        ### Create final array
        lines = np.zeros(len(self.Ebins)-1)

        ### Iterate over line list
        for k in range(len(e)):
            lines[insert_idx[k]] += emiss[k]
        
        #Return final array weighted by the norm and the 1e14 convention.
        return lines*1e14*self.norm
    
if __name__ == "__main__" :
    """
    ### NO LINES TEST ####################################
    
    ### XSPEC Spectrum ###
    
    AllModels.clear()
    AllData.clear()
            
    AllData.dummyrsp(0.2,12,10000, 'lin', 0.1, 0.00099 )
    
    m = Model("nlapec")
    m.setPars(6,0.6,0,2.3e-2)
    m.show()
    
    #Xset.abund = (np.ones(30)*0.8).tolist()
    #Xset.abund = 'angr'

    ebins = np.array(m.energies(0))
    elist = (ebins[1:] + ebins[:-1])/2
    
    spec_pyxspec = np.array(m.values(0))
    
    ### My APEC Spectrum ###
    
    apec = APEC()
    spec = apec(ebins, 6, 0.6, 2.3e-2)
    
    ### Comparison ###
    
    plt.figure(figsize = (10,10))
    plt.loglog(elist, spec_pyxspec)
    plt.loglog(elist, spec, ls = 'dashdot')
    plt.grid()
    plt.xlabel('Energy [keV]', fontsize = 15)
    plt.ylabel('Counts [ph/cm3/s]', fontsize = 15)
    plt.legend(('PyXSPEC', 'My APEC'), fontsize = 15)
    
    
    ### Relative error ###
    
    rel_err = (spec-spec_pyxspec)/spec_pyxspec
    plt.figure(figsize = (16,6))
    plt.loglog(elist, rel_err)
    plt.grid()
    plt.xlabel('Energy [keV]', fontsize = 15)
    plt.ylabel('Relative error', fontsize = 15)
    """
    ### WITH LINES ######################################

    ### XSPEC Spectrum ###
    
    AllModels.clear()
    AllData.clear()
            
    AllData.dummyrsp(0.2,12,10000, 'log', 0.1, 0.00099 )
    
    m = Model("apec")
    m.setPars(3,0.6,0,2.3e-2)
    m.show()
    
    #Xset.abund = (np.ones(30)*0.8).tolist()
    #Xset.abund = 'angr'

    ebins = np.array(m.energies(0))
    elist = (ebins[1:] + ebins[:-1])/2
    
    spec_pyxspec = np.array(m.values(0))
    
    ### My APEC Spectrum ###
    
    apec = APEC()
    spec = apec(ebins, 3, 0.6, 2.3e-2, addpseudo = True, addlines = True)
    
    ### Comparison ###
    
    plt.figure(figsize = (10,10))
    plt.loglog(elist, spec_pyxspec)
    plt.loglog(elist, spec, ls = 'dashdot')
    plt.grid()
    plt.xlabel('Energy [keV]', fontsize = 15)
    plt.ylabel('Counts [ph/cm3/s]', fontsize = 15)
    plt.legend(('PyXSPEC', 'My APEC'), fontsize = 15)
    plt.savefig('TestAPEC.pdf')
    
    ### Relative error ###
    
    rel_err = (spec-spec_pyxspec)/spec_pyxspec
    plt.figure(figsize = (16,6))
    plt.loglog(elist, rel_err)
    plt.grid()
    plt.xlabel('Energy [keV]', fontsize = 15)
    plt.ylabel('Relative error', fontsize = 15)
    