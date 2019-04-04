

def vmf_init(self, mu=None, kappa=None, mu_0=None, C_0=None, m_0=None, 
             sigma_0=None):
        """vmf init mock

        vonMisesFisherLogNormal.__init__ mock ment to fix the 
        compatability problem with Python 3.6.
        """
        self.__dict__ = {k:v for k,v in locals().items() if k != 'self'}

        self.mu_mf = self.mu_0
        self.C_mf = self.C_0

        self._gamma = self.C_mf 
        self._psi = self.mu_mf 
        
        checker = lambda x: True if not isinstance(x, type(None)) else None
        if (mu, kappa) == (None, None) and None not in map(checker, (mu_0, C_0, m_0, sigma_0)):
            self.resample()
