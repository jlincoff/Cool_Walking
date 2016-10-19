# Cool_Walking

Cool Walking (CW) is an enhanced sampling method originally developed in 2002: http://onlinelibrary.wiley.com/doi/10.1002/jcc.10181/abstract
This package contains codes for applying CW methods to peptides, as described in (JCP paper)
It also includes codes for standard replica exchange (REx) simulations, against which CW was compared
CW achieves equivalent or greater rates of convergence to the equilibrium distribution in comparison to REx, for greatly decreased cost
This is primarily because CW fully simulates only two replicas, as opposed to the dozens used in REx, and then applies a statistical annealing or cooling procedure during exchange attempts to maintain high acceptance ratios

There are two versions of each method: one that modifies the system temperature (TCW, TREx), and one that modifies the dielectric constant (by scaling of the charges) of the peptide (DCW, CREx)

Initial testing is set up for alanine dipeptide
