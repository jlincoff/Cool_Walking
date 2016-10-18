# This script sets up and performs a Coulomb replica exchange simulation.
# James Lincoff, October 2016

import math
import random
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

# Input pdb file, specify force field and temperature, solvate peptide.
pdb = PDBFile('alanine_init.pdb')
forcefield = ForceField('amber99sb.xml','tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
temperature = 300.0
modeller.addSolvent(forcefield,model = 'tip3p', boxSize = Vec3(2.0,2.0,2.0)*nanometers)
beta = Quantity(-1.0 / (1.3806488 * 6.022 / 1000 * temperature),1/kilojoules_per_mole) # Used to calculate boltzmann weights.
n_atoms = 22 # Number of solute atoms for which charges will be modified.

# Array of dielectrics for replicas.
dielectrics = [0.69, 1.0, 1.78, 4.0, 16.0, 10000.0]
numdiels = len(dielectrics)

# atts and accs store numbers of attempted and accepted exchanges. sys_objs, int_objs, and sim_objs store system, integrator, and simulation objects. set1 and set2 store indices of replicas for performing exchange attempts. track is used to build set1 and set2, and also to switch between performing exchange attempts on pairs represented by set1 and set2 during propagation.
atts = [0] * (numdiels - 1)
accs = [0] * (numdiels - 1)
sys_objs = []
int_objs = []
sim_objs = []
set1 = []
set2 = []
track = True

# Generate simulation objects and equilibrate.
for i in range(numdiels):
    
    # Build up set1 and set2.
    if i - 1 >= 0:
        if track:
            set1.append(i)
            track = False
        else:
            set2.append(i)
            track = True

    s = 'Preparing system ' + repr(i)
    print s

    # Construct system object and add thermostat. Add to list.
    sys = 't' + repr(i) + '_sys'
    vars()['sys'] = forcefield.createSystem(modeller.topology, nonbondedMethod = Ewald,nonbondedCutoff = 0.95*nanometers, constraints=HBonds)
    vars()['sys'].addForce(AndersenThermostat(temperature*kelvin, 25/picosecond))
    sys_objs.append(vars()['sys'])

    # Construct integrator and add to list.
    integrator = 't' + repr(i) + '_int'
    vars()['integrator'] = VerletIntegrator(0.001*picoseconds)
    int_objs.append(vars()['integrator'])

    # Construct simulation, add to list, modify charges to appropriate dielectric, minimize, and set velocities.
    sim = 't' + repr(i) + '_sim'
    vars()['sim'] = Simulation(modeller.topology, vars()['sys'], vars()['integrator'])
    vars()['sim'].context.setPositions(modeller.positions)
    sim_objs.append(vars()['sim'])

    forces = [sys_objs[i].getForce(force_index) for force_index in range(sys_objs[i].getNumForces())]
    forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
    force = forces[0]
    for atom_index in range(n_atoms):
        [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
        force.setParticleParameters(atom_index, charge/pow(dielectrics[i],0.5), sigma, epsilon)
        force.updateParametersInContext(sim_objs[i].context)

    sim_objs[i].minimizeEnergy()
    sim_objs[i].context.setVelocitiesToTemperature(temperature*kelvin)

    # Add reporters. In this case reporters are only added to the low temperature replica.
    if i == 0:
        sim_objs[i].reporters.append(PDBReporter('CREx_alanine.pdb',100))
        sim_objs[i].reporters.append(StateDataReporter('CREx_alanine.txt',100,step = True, potentialEnergy = True, totalEnergy = True, temperature = True))

    # Equilibrate.
    s = 'Equilibrating system ' + repr(i)
    print s
    sim_objs[i].step(1000)

# Begin production.
for steps in range(50000):
    s = 'Performing step ' + repr(steps)
    print s
    
    # Propagate all replicas.
    for i in range(numdiels):
        sim_objs[i].step(1000)
    
    
    s = 'Attempting swap ' + repr(steps)
    print s

    # Select which set of pairs of replicas to perform exchange attempts on based on track.
    if track:
        set = set1
        track = False
    else:
        set = set2
        track = True

    # Attempt exchanges.
    for j in range(len(set)):
        
        # Select a pair and increment number of attempts.
        pair = set[j]
        atts[pair - 1] = atts[pair - 1] + 1
        
        # Store current states for that pair of replicas.
        state1 = sim_objs[pair - 1].context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        state2 = sim_objs[pair].context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        energy1 = state2.getPotentialEnergy() + state1.getPotentialEnergy()
        sim_objs[pair - 1].context.setState(state2)
        sim_objs[pair].context.setState(state1)
        energy2 = sim_objs[pair - 1].context.getState(getEnergy = True).getPotentialEnergy() + sim_objs[pair].context.getState(getEnergy = True).getPotentialEnergy()

        # Calculate acceptance probability.
        acc = math.exp(beta*(energy2 - energy1))
        s = repr(pair) + ' ' + repr(acc)
        print s
        
        # Attempt exchange.
        if acc < random.uniform(0.0,1.0):
            sim_objs[pair - 1].context.setState(state1)
            sim_objs[pair].context.setState(state2)
        else:
            accs[pair - 1] = accs[pair - 1] + 1
    print str(atts)
    print str(accs)