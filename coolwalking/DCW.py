# This program sets up and runs a dielectric cool walking simulation, as described in "Comparing Generalized Ensemble Methods for Sampling of Systems with Many Degrees of Freedom" James Lincoff, Sukanya Sasmal, and Teresa Head-Gordon, Journal of Chemical Physics 2016.
# All parameters are specified in first 109 lines. The authors suggest reading through the temperature cool walking script first.
# James Lincoff, October 2016

import math
import random
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from decimal import *

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision':'mixed'}

# Input initial structure, solvate, create forcefield, and set temperature in Kelvin.
pdb = PDBFile('alanine_init.pdb')
forcefield = ForceField('amber99sb.xml','tip3p.xml')
temperature = 300.0
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield,model = 'tip3p', boxSize = Vec3(2.0,2.0,2.0)*nanometers)
beta = Quantity(-1.0 / (1.3806488 * 6.022 / 1000 * temperature),1/kilojoules_per_mole) # used for evaluating boltzmann weights

# Specify schedule of dielectrics that will be used for unscreening attempts (where a conformation originating from the high dielectric walker is unscreened through the intermediate dielectrics before
# attempting to be imposed on the low dielectric walker) in array dielectrics. Specify schedule of dielectrics for screening attempts (the inverse) in array up_diels. The schedules need not be idential but
# must include the maximum and minimum dielectrics, with array dielectrics going high to low and array up_diels going low to high.
dielectrics = [4.840, 3.666, 2.875, 2.363, 2.040, 1.832, 1.679, 1.539, 1.385, 1.204, 1.0]
num_diels = len(dielectrics)
up_diels = [1.0, 1.204, 1.385, 1.539, 1.679, 1.832, 2.040, 2.363, 2.875, 3.666, 4.840]
up_num = len(up_diels)

# step_probability controls the amount of annealing performed at each intermediate temperature during a cool walking cycle. With 1 fs timestep, step_probability = 0.025 gives on average 40 fs of
# equilibration per intermediate temperature. The authors propose 25 - 50 fs of equilibration per dielectric, increasing with system size, to balance the minimization of computational expense with the
# need for sufficient equilibration at each dielectric in order to achieve reasonable exchange rates.
step_probability = 0.04
# exchange_prob controls when, during a cool walking cycle, an actual exchange is attempted. During cooling of a high dielectric replica configuration, a random deviate is generated once equilibration
# at every dielectric temperature is complete (as controlled by step_probability). When this random deviate is less than exchange_prob, cooling of that high dielectric configuration is paused, and
# the equal and opposite heating cycle of a low dielectric configuration is begun at the current intermediate dielectric. exchange_prob of 0.025 gives an average of one exchange attempt every 40
# "levels" or dielectric. exchange_prob should be set such that an exchange is performed every cool walking cycle, so as not to waste computational effort. However, if exchange_prob is too high,
# there will be a waste of computational effort in the heating cycles performed for exchanges that are not likely to be accepted. The authors propose a value of exchange_prob slightly greater than
# 1 / (num_diels) as generally effective.
exch_prob = 0.1111
# decorrelation_num controls the amount of additional high dielectric replica decorrelation that is performed during unscreening. The number is a simple ratio of the number of steps of high dielectric
# replica propagation to the number of steps of unscreening that are performed on the high dielectric replica configurations used in exchange attempts. The authors propose a value of 50-100, increasing with
# system size, is appropriate to optimize the rate of convergence relative to the computational expense.
decorrelation_num = 50
# sim_length = total number of steps of propagation desired for production.
sim_length = 50000000
# unscreen_freq and screen_freq control the number of steps, on average, between exchange attempts for unscreening and screening.
unscreen_freq = 2000
screen_freq = 2000
# prop_steps dictates the amount of propagation performed at once for the high dielectric and low dielectric walker.
prop_steps = 100
num_steps = sim_length/prop_steps
p_up = prop_steps/screen_freq
p_down = prop_steps/unscreen_freq

# Create system objects and add thermostats.
low_diel = forcefield.createSystem(modeller.topology, nonbondedMethod = Ewald,nonbondedCutoff = 0.95*nanometers, constraints=HBonds)
low_diel.addForce(AndersenThermostat(temperature*kelvin, 50/picosecond))
high_diel = forcefield.createSystem(modeller.topology, nonbondedMethod = Ewald,nonbondedCutoff = 0.95*nanometers, constraints=HBonds)
high_diel.addForce(AndersenThermostat(temperature*kelvin, 50/picosecond))
annealer = forcefield.createSystem(modeller.topology, nonbondedMethod = Ewald,nonbondedCutoff = 0.95*nanometers, constraints=HBonds)
annealer.addForce(AndersenThermostat(temperature*kelvin, 50/picosecond))
probability_eval = forcefield.createSystem(modeller.topology, nonbondedMethod = Ewald,nonbondedCutoff = 0.95*nanometers, constraints=HBonds)
probability_eval.addForce(AndersenThermostat(temperature*kelvin, 50/picosecond))

# Create integrators.
low_integrator = VerletIntegrator(0.001*picoseconds)
high_integrator = VerletIntegrator(0.001*picoseconds)
anneal_integrator = VerletIntegrator(0.001*picoseconds)
probability_eval_integrator = VerletIntegrator(0.001*picoseconds)

# Create simulation objects
low_sim = Simulation(modeller.topology, low_diel, low_integrator, platform, properties)
low_sim.context.setPositions(modeller.positions)
high_sim = Simulation(modeller.topology, high_diel, high_integrator, platform, properties)
high_sim.context.setPositions(modeller.positions)
anneal_sim = Simulation(modeller.topology, annealer, anneal_integrator, platform, properties)
anneal_sim.context.setPositions(modeller.positions)
p_eval_sim = Simulation(modeller.topology, probability_eval, probability_eval_integrator, platform, properties)
p_eval_sim.context.setPositions(modeller.positions)

# Modify charges on high dielectric replica.
n_atoms = 22 # Number of solute atoms for which charges will be modified.
standard_charges = list()
forces = [high_diel.getForce(force_index) for force_index in range(high_diel.getNumForces())]
forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
force = forces[0]
for atom_index in range(n_atoms):
    [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
    standard_charges.append(charge)
    force.setParticleParameters(atom_index, charge/pow(dielectrics[0],0.5), sigma, epsilon)
    force.updateParametersInContext(high_sim.context)

# Minimize energies.
low_sim.minimizeEnergy()
high_sim.minimizeEnergy()

# Set velocities.
low_sim.context.setVelocitiesToTemperature(temperature*kelvin)
high_sim.context.setVelocitiesToTemperature(temperature*kelvin)
# Add in reporters. countsfile tracks acceptances.
low_sim.reporters.append(PDBReporter('DCW_alanine_test.pdb', 1000))
low_sim.reporters.append(StateDataReporter('DCW_alanine_test.txt', 1000, step=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True))
countsfile = open('DCW_alanine_test_counts.txt','w')
# Equilibrate.
low_sim.step(500000)
high_sim.step(500000)

# These track number of attempted and accepted exchange attempts.
att_up = 0
att_down = 0
acc_up = 0
acc_down = 0

# Begin production.
for steps in range(num_steps):
    
    # Propagate replicas.
    low_sim.step(prop_steps)
    high_sim.step(prop_steps)
    
    # Evaluate unscreening probability
    if random.uniform(0.0,1.0) <= p_down:
        
        # Track exchange attempts and store current configurations.
        att_down = att_down + 1
        initial_high_state = high_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        initial_low_state = low_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        
        # Prepare for unscreening.
        forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
        forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
        force = forces[0]
        for atom_index in range(n_atoms):
            [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
            force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(dielectrics[0],0.5), sigma, epsilon)
            force.updateParametersInContext(anneal_sim.context)
        forces = [probability_eval.getForce(force_index) for force_index in range(probability_eval.getNumForces())]
        forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
        force = forces[0]
        for atom_index in range(n_atoms):
            [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
            force.setParticleParameters(atom_index, standard_charges[atom_index], sigma, epsilon)
            force.updateParametersInContext(p_eval_sim.context)
        anneal_sim.context.setState(initial_high_state)
        Tcwdown = 1.0

        # Begin unscreening.
        for cool_step in range(1,num_diels,1):
            
            # Move annealer down by one dielectric and capture probability for that unscreening move, Tcwdown.
            energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
            forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
            force = forces[0]
            for atom_index in range(n_atoms):
                [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(dielectrics[cool_step],0.5), sigma, epsilon)
                force.updateParametersInContext(anneal_sim.context)
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            Tcwdown = Tcwdown * math.exp(beta*(energy2 - energy1))

            # Prepare to propagate and collect window of states data.
            W_down = 1.0
            anneal_num = 0
            anneal_sim.step(1)
            energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            while random.uniform(0.0,1.0) >= step_probability:
                anneal_sim.step(1)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                W_down = W_down + math.exp(beta*(energy2-energy1))
                
                # Perform additional propagation of high dielectric walker to decorrelate.
                high_sim.step(decorrelation_num)
                anneal_num = anneal_num + 1
            
            # Determine whether or not to exchange at this intermediate dielectric.
            if random.uniform(0.0,1.0) >= exch_prob:
                continue
            
            # If so, prepare for exchange by storing current configuratino and prepare to perform equal and opposite screening procedure to balance the unscreening that was just performed.
            cooled_state = anneal_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
            Tcwup = 1.0
            anneal_sim.context.setState(initial_low_state)
            
            # Collect window of states data.
            W_up = 0.0
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            W_up = W_up + math.exp(beta*(energy2-energy1))
            for anneal_step in range(anneal_num):
                anneal_sim.step(1)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                W_up = W_up + math.exp(beta*(energy2-energy1))
            
            # Perform screening, capture probability for performing screening Tcwup.
            for heat_step in range(cool_step, 0, -1):
                forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
                forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
                force = forces[0]
                for atom_index in range(n_atoms):
                    [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                    force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(dielectrics[heat_step-1],0.5), sigma, epsilon)
                    force.updateParametersInContext(anneal_sim.context)
                while random.uniform(0.0,1.0) >= step_probability:
                    anneal_sim.step(1)
                energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
                forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
                force = forces[0]
                for atom_index in range(n_atoms):
                    [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                    force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(dielectrics[heat_step],0.5), sigma, epsilon)
                    force.updateParametersInContext(anneal_sim.context)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                Tcwup = Tcwup * math.exp(beta*(energy2 - energy1))
    
            # Check that neither probability is 0 to avoid math error.
            if Tcwup == 0 or W_up == 0:
                break
            forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
            forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
            force = forces[0]
            for atom_index in range(n_atoms):
                [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(dielectrics[cool_step],0.5), sigma, epsilon)
                force.updateParametersInContext(anneal_sim.context)
            
            # Calculate total acceptance probability.
            acc = Tcwdown/Tcwup*W_down/W_up
            p_eval_sim.context.setState(cooled_state)
            anneal_sim.context.setState(initial_low_state)
            energy1 = p_eval_sim.context.getState(getEnergy = True).getPotentialEnergy()
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            ener = energy1 + energy2
            p_eval_sim.context.setState(initial_low_state)
            anneal_sim.context.setState(cooled_state)
            energy1 = p_eval_sim.context.getState(getEnergy = True).getPotentialEnergy()
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            acc = acc * math.exp(beta*(ener - energy1 - energy2))
            print s
            
            # Attempt exchange.
            if (acc > random.uniform(0.0,1.0)):
                acc_down = acc_down + 1
                low_sim.context.setState(cooled_state)
                s = 'for ' + repr(steps) + ' att_down ' + repr(att_down) + ' cooling step ' + repr(cool_step) +  ' acc ' + repr(acc) + ' acc_down ' + repr(acc_down)
                countsfile.write(s)
                break

    # Evaluate screening probability, for taking a conformation from the low dielectric replica and applying to the high dielectric replica. This consists of the equal and opposite procedure of the unscreening
    # attempt directly above, though without additional decorrelation of the high or low dielectric replicas.
    elif random.uniform(0.0,1.0) <= p_up/(1-p_down):
        att_up = att_up + 1
        initial_high_state = high_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        initial_low_state = low_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
        forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
        force = forces[0]
        for atom_index in range(n_atoms):
            [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
            force.setParticleParameters(atom_index, standard_charges[atom_index], sigma, epsilon)
            force.updateParametersInContext(anneal_sim.context)
        forces = [probability_eval.getForce(force_index) for force_index in range(probability_eval.getNumForces())]
        forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
        force = forces[0]
        for atom_index in range(n_atoms):
            [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
            force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(dielectrics[0],0.5), sigma, epsilon)
            force.updateParametersInContext(p_eval_sim.context)
        anneal_sim.context.setState(initial_low_state)
        Tcwup = 1.0
        for heat_step in range(1,up_num,1):
            energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
            forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
            force = forces[0]
            for atom_index in range(n_atoms):
                [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(up_diels[heat_step],0.5), sigma, epsilon)
                force.updateParametersInContext(anneal_sim.context)
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            Tcwup = Tcwup * math.exp(beta*(energy2 - energy1))
            W_up = 1.0
            anneal_num = 0
            anneal_sim.step(8)
            energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            while random.uniform(0.0,1.0) >= step_probability:
                anneal_sim.step(1)
                anneal_num = anneal_num + 1
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                W_up = W_up + math.exp(beta*(energy2-energy1))
            if random.uniform(0.0,1.0) >= exch_prob:
                continue
            heated_state = anneal_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
            Tcwdown = 1.0
            anneal_sim.context.setState(initial_high_state)
            W_down = 0.0
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            W_down = W_down + math.exp(beta*(energy2-energy1))
            for anneal_step in range(anneal_num):
                anneal_sim.step(1)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                W_down = W_down + math.exp(beta*(energy2-energy1))
            for cool_step in range(heat_step, 0, -1):
                forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
                forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
                force = forces[0]
                for atom_index in range(n_atoms):
                    [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                    force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(up_diels[cool_step-1],0.5), sigma, epsilon)
                    force.updateParametersInContext(anneal_sim.context)
                while random.uniform(0.0,1.0) >= step_probability:
                    anneal_sim.step(1)
                energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
                forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
                force = forces[0]
                for atom_index in range(n_atoms):
                    [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                    force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(up_diels[cool_step],0.5), sigma, epsilon)
                    force.updateParametersInContext(anneal_sim.context)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                Tcwdown = Tcwdown * math.exp(beta*(energy2 - energy1))
            if Tcwdown == 0 or W_down == 0:
                break
            forces = [annealer.getForce(force_index) for force_index in range(annealer.getNumForces())]
            forces = [force for force in forces if isinstance(force, openmm.NonbondedForce)]
            force = forces[0]
            for atom_index in range(n_atoms):
                [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                force.setParticleParameters(atom_index, standard_charges[atom_index]/pow(up_diels[heat_step],0.5), sigma, epsilon)
                force.updateParametersInContext(anneal_sim.context)
            acc = Tcwup/Tcwdown*W_up/W_down
            p_eval_sim.context.setState(heated_state)
            anneal_sim.context.setState(initial_high_state)
            energy1 = p_eval_sim.context.getState(getEnergy = True).getPotentialEnergy()
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            ener = energy1 + energy2
            p_eval_sim.context.setState(initial_high_state)
            anneal_sim.context.setState(heated_state)
            energy1 = p_eval_sim.context.getState(getEnergy = True).getPotentialEnergy()
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            acc = acc * math.exp(beta*(ener - energy1 - energy2))
            if (acc > random.uniform(0.0,1.0)):
                acc_up = acc_up + 1
                high_sim.context.setState(heated_state)
                s = 'for ' + repr(steps) + ' att_up ' + repr(att_up) + ' heating step ' + repr(heat_step) +  ' acc ' + repr(acc) + ' acc_up ' + repr(acc_up)
                countsfile.write(s)
                break
countsfile.close()
