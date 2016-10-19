# This program sets up and runs a temperature cool walking simulation, as described in "Comparing Generalized Ensemble Methods for Sampling of Systems with Many Degrees of Freedom" James Lincoff, Sukanya Sasmal, and Teresa Head-Gordon, Journal of Chemical Physics 2016.
# All parameters are specified in first 101 lines, with exception of names of restart files saved in lines 115 and 116.
# James Lincoff, October 2016

import math
import random
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from decimal import *

beta = Quantity(-1.0 / (1.3806488 * 6.022 / 1000.0), 1/kilojoules_per_mole) # used for evaluating Boltzmann weights
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision':'mixed'}

# Input system files.
pdb = PDBFile('alanine_init.pdb')

# Input array of temperatures in Kelvin, low to high. The lowest temperature is the temperature of the target replica. The highest temperature is the temperature of the high-sampling replica. The
# intermediate temperatures form the annealing schedule through which high temperature configurations are cooled during exchanges.
temperatures = [300.0, 314.5, 331.0, 347.5, 367.0, 387.5, 410.0]
num_temps = len(temperatures) # used in control of annealing cycles

# Create forcefield and System objects. Use Modeller to solvate. low_temp is the replica of interest, high_temp is the high-sampling replica, and annealer is used to perform annealing during exchange
# attempts. The same naming scheme is used for Integrator objects and Simulation objects. Add thermostats to low_temp and high_temp.
forcefield = ForceField('amber99sb.xml','tip3p.xml')
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model = 'tip3p', boxSize = Vec3(2.0,2.0,2.0)*nanometers)
low_temp = forcefield.createSystem(modeller.topology,nonbondedMethod = Ewald,nonbondedCutoff = 0.9*nanometers, constraints=HBonds)
low_temp.addForce(AndersenThermostat(temperatures[0],25/picosecond))
high_temp = forcefield.createSystem(modeller.topology,nonbondedMethod = Ewald,nonbondedCutoff = 0.9*nanometers, constraints=HBonds)
high_temp.addForce(AndersenThermostat(temperatures[num_temps-1],25/picosecond))
annealer = forcefield.createSystem(modeller.topology,nonbondedMethod = Ewald,nonbondedCutoff = 0.9*nanometers, constraints=HBonds)

# Create integrators.
low_integrator = VerletIntegrator(0.001*picoseconds)
high_integrator = VerletIntegrator(0.001*picoseconds)
anneal_integrator = VerletIntegrator(0.001*picoseconds)

# Create simulations and set initial positions to that of the inputted structure.
low_sim = Simulation(modeller.topology, low_temp, low_integrator, platform, properties)
low_sim.context.setPositions(modeller.positions)
high_sim = Simulation(modeller.topology, high_temp, high_integrator, platform, properties)
high_sim.context.setPositions(modeller.positions)
anneal_sim = Simulation(modeller.topology, annealer, anneal_integrator, platform, properties)
anneal_sim.context.setPositions(modeller.positions)

# Set probabilistic controls for cool walking cycles.
# p_down controls when, during normal propagation of the two replicas, a cool walking cycle is initiated. With 1 fs timestep, p_down = 0.00025 gives on average one cool walking attempt per 4 ps of
# normal propagation. The authors propose p_down be set to give a cool walking attempt every few ps, so as to allow for an exchanged conformation sufficient time to propagate and explore phase space
# before being replaced by another.
p_down = 0.0005
# step_probability controls the amount of annealing performed at each intermediate temperature during a cool walking cycle. With 1 fs timestep, step_probability = 0.025 gives on average 40 fs of
# equilibration per intermediate temperature. The authors propose 25 - 50 fs of equilibration per temperature, increasing with system size, to balance the minimization of computational expense with the
# need for sufficient equilibration at each temperature in order to achieve reasonable exchange rates.
step_probability = 0.04
# exchange_prob controls when, during a cool walking cycle, an actual exchange is attempted. During cooling of a high temperature replica configuration, a random deviate is generated once equilibration
# at every intermediate temperature is complete (as controlled by step_probability). When this random deviate is less than exchange_prob, cooling of that high temperature configuration is paused, and
# the equal and opposite heating cycle of a low temperature configuration is begun at the current intermediate temperature. exchange_prob of 0.025 gives an average of one exchange attempt every 40
# "levels" or temperatures. exchange_prob should be set such that an exchange is performed every cool walking cycle, so as not to waste computational effort. However, if exchange_prob is too high,
# there will be a waste of computational effort in the heating cycles performed for exchanges that are not likely to be accepted. The authors propose a value of exchange_prob slightly greater than
# 1 / (num_temps) as generally effective.
exchange_prob = 0.2
# decorrelation_num controls the amount of additional high temperature replica decorrelation that is performed during annealing. The number is a simple ratio of the number of steps of high temperature
# replica propagation to the number of steps of cooling that are performed on the high temperature replica configurations used in exchange attempts. The authors propose a value of 5-10, increasing with
# system size, is appropriate to optimize the rate of convergence relative to the computational expense.
decorrelation_num = 8
# performWOS controls whether the window of states procedure is applied. WOS weighting to the acceptance probability increases the rate of convergence for a given amount of data generated, but slows the
# rate of data production because it requires additional CPU-GPU communication. This effect will be larger for increased system sizes.
performWOS = True

# Minimize energies. anneal_sim is not minimized, as all propagation performed by anneal_sim will be done on conformations that are sourced from low_sim and high_sim.
low_sim.minimizeEnergy()
high_sim.minimizeEnergy()

# Set velocities. anneal_sim again does not require initial velocities.
low_sim.context.setVelocitiesToTemperature(temperatures[0]*kelvin)
high_sim.context.setVelocitiesToTemperature(temperatures[num_temps-1]*kelvin)

# Add reporters as desired. "countsfile" is used to track exchange acceptance rates.
low_sim.reporters.append(PDBReporter('TCW_alanine_test.pdb', 1000))
low_sim.reporters.append(StateDataReporter('TCW_alanine_test.txt', 1000, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))
countsfile = open('TCW_alanine_test_counts.txt','w')
att_down = 0 # tracks number of attempted cool walking cycles
acc_down = 0 # tracks number of accepted cool walking exchanges

# Propagate replicas to equilibrate.
low_sim.step(500000)
high_sim.step(500000)

# totstep is used to track simulation progress. whetherToSave will be used to determine when restart files are generated. checkpoint_constant is the save frequency (in this case every nanosecond).
# checkpointCounter is used to determine when to save.
totstep = 0
whetherToSave = False
checkpoint_constant = 1000000
checkpointCounter = 1000000


# Begin production run. Set the number of total steps of low temperature replica propagation as desired.
while totstep < 50000000:
    
    # Always perform at least one step. Then generate a series of random deviates, comparing to p_down, to determine how much propagation will occur before performing a cooling trajectory. Perform all of this
    # propagation at once to save on CPU-GPU communication. If totstep exceeds checkpointCounter, change whetherToSave to true to save a checkpoint.
    totstep = totstep + 1
    num_steps = 1
    while random.uniform(0.0,1.0) >= p_down:
        totstep = totstep + 1
        num_steps = num_steps + 1
    if totstep >= checkpointCounter:
        whetherToSave = True
    low_sim.step(num_steps)
    high_sim.step(num_steps)
    if whetherToSave:
        low_sim.saveState('TCW_alanine_test_low_st'+str(checkpointCounter/checkpoint_constant)+'.xml')
        high_sim.saveState('TCW_alanine_test_high_st'+str(checkpointCounter/checkpoint_constant)+'.xml')
        checkpointCounter = checkpointCounter + checkpoint_constant
        whetherToSave = False

    
    # Now begin cooling trajectory. Track number of exchange attempts.
    att_down = att_down + 1
    
    # Store current configurations of high and low temperature replicas, set anneal_sim to high temperature replica state for cooling, and initialize value of transition probability for cooling,
    # Tcwdown. Store energies and velocities that will be used later.
    initial_high_state = high_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
    initial_low_state = low_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
    initial_PE = initial_low_state.getPotentialEnergy()
    initial_KE = initial_low_state.getKineticEnergy()
    initial_velocities = initial_low_state.getVelocities()
    anneal_sim.context.setState(initial_high_state)
    Tcwdown = 0.0

    # Begin cooling cycle. Will anneal starting at the second highest temperature and continue down to the second lowest temperature in the schedule.
    for cool_step in range(1,num_temps,1):
        
        # Store the energy of the current configuration (energy1) for use in generated Tcwdown, then change temperature to the next (immediately lower) temperature in the schedule for propagation.
        # Store the new energy (energy2), and update Tcwdown.
        PE1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
        KE1 = anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
        energy1 = PE1 + KE1
        anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities = True).getVelocities()*math.sqrt(temperatures[num_temps-cool_step-1]/temperatures[num_temps - cool_step]))
        energy2 = PE1 + KE1*temperatures[num_temps-cool_step-1]/temperatures[num_temps - cool_step]
        Tcwdown = Tcwdown + beta*(energy2/temperatures[num_temps-cool_step-1]-energy1/temperatures[num_temps-cool_step])
        

        # If performing WOS weighting:
        if performWOS:
            # Prepare for propagation and collection of window of states data. Window of states weightings consist of sums of Boltzmann factors, and so must be normalized in a different way than
            # transition probabilities and Metropolis exchange probabilities, which are products of Boltzmann factors, to avoid math overflow errors given large system energies. The energy from the
            # first step of propagation will be stored as energy1, and will be subtracted from every Boltzmann factor that goes into both W_down and W_up.
            anneal_num = 0 # Will track the number of steps performed while generating window of states data.
            anneal_sim.step(1) # First step of propagation.
            high_sim.step(decorrelation_num) # High temperature decorrelation for first step.
            energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            W_vals = list()
            
            # Perform propagation and collect window of states weighting. For every step, perform high temperature decorrelation, collect energy of current annealed configuration, and add into window
            # of states weighting.
            while random.uniform() >= step_probability:
                anneal_sim.step(1)
                high_sim.step(decorrelation_num)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                number = str(beta*(energy2-*energy1)/temperatures[num_temps-cool_step-1])
                W_vals.append(number) # Store exponents for Boltzmann factors for now to reduce number of Decimal precision calculations that are required.
                anneal_num = anneal_num + 1

        # If not performing window of states weighting. Performing all propagation at once increases simulation speed by reducing CPU-GPU communication.
        else:
            anneal_num = 0
            anneal_sim.step(1)
            high_sim.step(decorrelation_num)
            while random.uniform(0.0,1.0) >= step_probability:
                anneal_num = anneal_num + 1
            anneal_sim.step(anneal_num)
            high_sim.step(decorrelation_num*anneal_num)

        # Once propagation at the current temperature is complete, it must be decided whether an exchange attempt will be performed at the current temperature, or whether cooling will simply
        # continue.
        if (np.random.random_sample() > exchange_prob):
            continue

        # If it is decided to perform an exchange attempt at this temperature, prepare to perform a heating trajectory. The current state of anneal_sim is stored as cooled_state. This is the
        # configuration that we will attempt to impose on the low temperature replica.
        cooled_state = anneal_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        Tcwup = 0.0
        anneal_sim.context.setState(initial_low_state)
        anneal_sim.context.setVelocities(initial_low_state.getVelocities()*math.sqrt(temperatures[num_temps-cool_step-1]/temperatures[0]))
        # If performing window of states weighting:
        if performWOS:
            W_down = Decimal(1) # Will be used to store the window of states weighting for cooling to the current temperature.
            for i in range(len(W_vals)):
                W_down = W_down + Decimal(W_vals[i]).exp()
            
            # Prepare to collect a transition probability for heating (Tcwup) and a window of states weighting for the current temperature (W_up). Note that the first term in W_up consists of
            # evaluating the energy of initial_low_state at the intermediate temperature from which the exchange is attempted.
            W_up = Decimal(0)
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            number = str(beta*(energy2-*energy1)/temperatures[num_temps-cool_step-1])
            W_up = W_up + Decimal(number).exp()

            # Propagate at the current temperature, collecting data for W_up. For this temperature only, the number of annealing steps is controlled such that W_down and W_up have the same number of
            # Boltzmann factors.
            for anneal_step in range(anneal_num):
                anneal_sim.step(1)
                energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
                number = str(beta*(energy2-*energy1)/temperatures[num_temps-cool_step-1])
                W_up = W_up + Decimal(number).exp()
        else:
            anneal_sim.step(anneal_num)

        # Continue heating the configuration through the temperature schedule up to the maximum temperature, collecting data for Tcwup.
        for heat_step in range(num_temps-cool_step, num_temps, 1):
            anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities = True).getVelocities()*math.sqrt(temperatures[heat_step]/temperatures[heat_step-1]))
            anneal_num = 0
            while random.uniform() >= step_probability:
                anneal_num = anneal_num + 1
            anneal_sim.step(anneal_num)
            PE1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
            KE1 = anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
            energy1 = PE1 + KE1
            anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities = True).getVelocities()*math.sqrt(temperatures[heat_step-1]/temperatures[heat_step]))
            energy2 = PE1 + KE1*temperatures[heat_step-1]/temperatures[heat_step]
            Tcwup = Tcwup + beta*(energy2/temperatures[heat_step-1] - energy1/temperatures[heat_step])

        # Begin generating the value of the acceptance probability. This value acc is formally the natural logarithm of the acceptance probability. This is done to avoid math overflow errors,
        # resulting from the large energies of the system.
        if performWOS:
            acc = Decimal(str(Tcwdown-Tcwup))+Decimal(W_down/W_up).ln()
        else:
            acc = Decimal(str(Tcwdown-Tcwup))
        # Use low_sim and anneal_sim to obtain the energy weightings used for the Metropolis exchange criterion aspects of acc. Ensure that low_sim is set back to initial_low_state and anneal_sim
        # is set to cooled_state before evaluating the exchange probability, so that if the exchange is rejected annealing and propagation continue as normal.
        cooled_PE = cooled_state.getPotentialEnergy()
        cooled_KE = cooled_state.getKineticEnergy()
        energy1 = cooled_PE + cooled_KE*temperatures[0]/temperatures[num_temps-cool_step-1]
        energy2 = initial_PE + initial_KE*temperatures[num_temps-cool_step-1]/temperatures[0]
        ener = beta*energy1/temperatures[num_temps-cool_step-1] + beta*energy2/temperatures[0]
        low_sim.context.setState(initial_low_state)
        anneal_sim.context.setState(cooled_state)
        energy1 = initial_PE + initial_KE
        energy2 = cooled_PE + cooled_KE
        ener2 = beta*energy1/temperatures[0] + beta*energy2/temperatures[num_temps-cool_step-1]

        # Obtain final value of acc, and generate acc_prob, the random deviate against which acc is compared.
        acc = acc + Decimal(str(ener-ener2))
        acc_prob = random.uniform()

        # If acc_prob = 0, the swap is accepted no matter what. Use this first if statement to avoid taking ln(0). Break out of the cooling cycle and return to normal propagation.
        if acc_prob == 0.0:
            acc_down = acc_down + 1 # Track the number of accepted exchanges.
            low_sim.context.setState(cooled_state) # Set low_sim to the cooled configuration.
            low_sim.context.setVelocities(cooled_state.getVelocities()*math.sqrt(temperatures[0]/temperatures[num_temps - cool_step -1])) # Scale velocities to appropriate value.
            
            # Collect data on exchanges, in this case when and where exchanges are accepted.
            s = repr(totstep) + ' ' + repr(att_down) + ' ' + repr(cool_step) + ' ' + repr(acc) + ' '+ repr(acc_down) + '\n'
            countsfile.write(s)
            break

        # If acc_prob is not one, perform the usual comparison. If the exchange is rejected, the cooling cycle will continue until either an exchange is accepted or the entire temperature schedule
        # is run through, at which point normal propagation is restarted.
        elif (acc > Decimal(str(acc_prob)).ln()):
            acc_down = acc_down + 1
            low_sim.context.setState(cooled_state)
            low_sim.context.setVelocities(cooled_state.getVelocities()*math.sqrt(temperatures[0]/temperatures[num_temps - cool_step -1]))
            s = repr(totstep) + ' ' + repr(att_down) + ' ' + repr(cool_step) + ' ' + repr(acc) + ' ' + repr(acc_down) + '\n'
            countsfile.write(s)
            break
countsfile.close()
