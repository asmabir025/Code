#!/bin/bash

Namid="2048" 
Nbmid=$Namid

# Defining our change in timestep. We have tested this, 0.01 seems work the best.
Deltatcreate="0.01" 
DeltatMDshear="0.01" 

# Here we are defining the total amount of strain on our system, separated by elastic, transient, equillibrium, and eq_infinity stage.  
elastic_strain_max="0.0005" 
transient_strain_max="0.0995" 
eq_strain_max="0.4"
inf_strain_max="1.4"
# Here we define the total number of data files 
dump_elastic_freq="0.00001" 
dump_transient_freq="0.000995" 
dump_eq_freq="0.001" 
dump_inf_freq="0.001" 

# Here we define the total number of data points we want to take for each region
thermo_elastic_freq="0.000001"
thermo_transient_freq="0.0000995" 
thermo_eq_freq="0.00045" 
thermo_inf_freq="0.00045"

# Defining strain frequencies for chunck averging in equilibrium and infinity regions
chunk_eq_freq="0.01" 
chunk_inf_freq="0.0001"
# Defining strain frequencies for Nevery and Nrepeat
Nevery_inf="0.00001"
Nevery_eq="0.001"
Nrepeat="10"


# Defining arrays to create our directory tree
declare -a Pinsarray
declare -a Phiarray
declare -a gammadotearray 
declare -a gammadotbeforepointarray
# declare -a regioname
 

# Defining Pins array
Pinsarray=(9)

# Defining phi array
Phiarray=(0.845) #changed
# Defining coefficent for gammadot
gammadotbeforepointarray=(1)
# Defining the power for gammadot
gammadotearray=(6) #changed

random_num1=(1 15 157 1508 17789 156098 1435248 10948712 187236095 1439823604)
random_num2=(7 85 907 4309 67789 556798 5437249 40944711 584236065 9439423602)


# Directory tree: Config -> Pins -> Phi -> gammadot 
for index in "${!random_num1[@]}"
     do 
     dirname01="Config""$((${index}+1))"
     mkdir -p $dirname01
     for Pins in "${Pinsarray[@]}"
     do
     dirname=$dirname01/"Pins""${Pins}"
     mkdir -p $dirname
     for Phi in "${Phiarray[@]}"
     do
          Phitimes1many0=($(bc -l <<< "$Phi * 100000"))
          Phistringprel=$(printf "%.0f" $Phitimes1many0)
          Phistring=$(printf "%06d" $Phistringprel)
          dirname0=$dirname/"Phi"$Phistring
          mkdir -p $dirname0
          for gammadotbeforepoint in "${gammadotbeforepointarray[@]}"
          do
          for gammadote in "${gammadotearray[@]}"
          do
               gammadot=($(bc -l <<< "$gammadotbeforepoint * 10^(-$gammadote)"))
               dirgammadotname="gammadot"$gammadotbeforepoint"eminus"$gammadote
               dirname1=$dirname/"Phi"$Phistring/$dirgammadotname
               # rm -R $dirname1
               mkdir -p $dirname1
               
     # Copying over the lmp script to run lammps
               cp /work2/01373/abug1/stampede2/lmp_Eshan_stampede $dirname1
          
               # Creating a restart and dump file directory to store output too
               dirname2=$dirname1/"Restart_Files"
               mkdir -p $dirname2
               dirname3=$dirname1/"Dump_Files"
               mkdir -p $dirname3
          
               # Creating our input script
               input_create="input_Npin"$Pins"_Phi"$Phistring"_gammadot"$gammadotbeforepoint"eminus"$gammadote"_create"

               echo ' '> $dirname1/$input_create

               # Defining our units -> Lennard-Jones
               echo 'units lj' >> $dirname1/$input_create

               # Defining our dimensions. Here, we use two dimensions 
               echo 'dimension 2' >> $dirname1/$input_create

               # Defining our atom style. atomic style is good for course-grained solids, liquids, and metals.
               echo 'atom_style atomic' >> $dirname1/$input_create

               # Defining a modification to the atom. We need this line to track single particle trajectories.  
               echo 'atom_modify map array' >> $dirname1/$input_create

               # Defining boundary conditions. We initially begin with periodic boundary conditions on all sides of the box.
               echo 'boundary p p p' >> $dirname1/$input_create

               # Creating variables for our timestep, temperature, strain rate, gamma value for the dpd pair_style, number of particles, packing fraction, radii size, cutoff distance(Daa, Dab, etc.), A value for dpd pair_style, length of the box, size of the walls, and unit cell size to create the lattice. 
               echo 'variable tstep equal' $Deltatcreate >> $dirname1/$input_create
               echo 'variable Tempr equal 0.0' >> $dirname1/$input_create
               echo 'variable gammadoterate equal' $gammadot  >> $dirname1/$input_create
               echo 'variable gammadpd equal 1.0' >> $dirname1/$input_create
               echo 'variable gammadpdaa equal ${gammadpd}' >> $dirname1/$input_create
               echo 'variable gammadpdab equal ${gammadpd}' >> $dirname1/$input_create
               echo 'variable gammadpdbb equal ${gammadpd}' >> $dirname1/$input_create
               echo 'variable gammadpdac equal ${gammadpd}' >> $dirname1/$input_create
               echo 'variable gammadpdbc equal ${gammadpd}' >> $dirname1/$input_create
               echo 'variable gammadpdcc equal ${gammadpd}' >> $dirname1/$input_create
               echo 'variable Namid equal' $Namid >> $dirname1/$input_create
               echo 'variable Nbmid equal' $Nbmid >> $dirname1/$input_create
               echo 'variable Ntotmid equal ${Namid}+${Nbmid}' >> $dirname1/$input_create
               echo 'variable phi equal' $Phi >> $dirname1/$input_create
               echo 'variable phiwall equal ${phi}' >> $dirname1/$input_create
               echo 'variable Ra equal 1.0' >> $dirname1/$input_create
               echo 'variable Rb equal 1.4' >> $dirname1/$input_create
               echo 'variable Rpin equal ${Ra}*(0.004)' >> $dirname1/$input_create
               echo 'variable eps equal 1.0' >> $dirname1/$input_create
               echo 'variable Daa equal ${Ra}+${Ra}' >> $dirname1/$input_create
               echo 'variable Dab equal ${Ra}+${Rb}' >> $dirname1/$input_create
               echo 'variable Dbb equal ${Rb}+${Rb}' >> $dirname1/$input_create
               echo 'variable Dac equal ${Ra}+${Rpin}' >> $dirname1/$input_create
               echo 'variable Dbc equal ${Rb}+${Rpin}' >> $dirname1/$input_create
               echo 'variable Dcc equal ${Rpin}+${Rpin}' >> $dirname1/$input_create
               echo 'variable Aaa equal ${eps}/${Daa}' >> $dirname1/$input_create
               echo 'variable Aab equal ${eps}/${Dab}' >> $dirname1/$input_create
               echo 'variable Abb equal ${eps}/${Dbb}' >> $dirname1/$input_create
               echo 'variable Aac equal ${eps}/${Dac}' >> $dirname1/$input_create
               echo 'variable Abc equal ${eps}/${Dbc}' >> $dirname1/$input_create
               echo 'variable Acc equal ${eps}/${Dcc}' >> $dirname1/$input_create
               echo 'variable Dmax equal ${Dbb}' >> $dirname1/$input_create
               echo 'variable Lmid equal sqrt(PI*((${Ra}^2)*${Namid}+(${Rb}^2)*${Nbmid})/${phi})' >> $dirname1/$input_create
               echo 'variable Npin equal' $Pins >> $dirname1/$input_create
               echo 'variable Nbasis equal 1' >> $dirname1/$input_create
               echo 'variable Npinx equal sqrt(${Npin})' >> $dirname1/$input_create
               echo 'variable Lywall equal  ${Dbb}*3.0' >> $dirname1/$input_create
               echo 'variable walltopystart equal ${Lywall}+${Lmid}' >> $dirname1/$input_create
               echo 'variable walltopyend equal 2.0*${Lywall}+${Lmid}' >> $dirname1/$input_create
               echo 'variable Lunitcellx equal ${Lmid}/${Npinx}' >> $dirname1/$input_create
               echo 'variable Lunitcelly equal ${walltopystart}/${Npinx}' >> $dirname1/$input_create
               echo 'variable LunitcellLywall equal ${Lywall}/${Npinx}' >> $dirname1/$input_create
               echo 'variable Lunitcelltopyend equal ${walltopyend}/${Npinx}' >> $dirname1/$input_create
               echo 'variable rhostar equal ${Nbasis}/${Lunitcellx}^2' >> $dirname1/$input_create
          #Amy comments out following line defining ybasis ... it is legacy from wrong way of doing pins 
               #echo 'variable ybasis equal 0.5+${Lywall}/${Lunitcellx}'>> $dirname1/$input_create

               # Defining our lattice. Here we are creating a square lattice.
               echo 'lattice custom ${rhostar} &' >> $dirname1/$input_create
               echo '        a1      1.0     0.0     0.0     &' >> $dirname1/$input_create
               echo '        a2      0.0     1.0     0.0     &' >> $dirname1/$input_create
               echo '        a3      0.0     0.0     1.0     &' >> $dirname1/$input_create
               echo '        basis   0.5     0.5     0.0' >> $dirname1/$input_create
               
               # Defining our regions. Namely, the mid_region and the top and bottom walls
               echo 'region top_region block 0 ${Lmid} ${walltopystart} ${walltopyend} -0.01 0.01 units box' >> $dirname1/$input_create
               echo 'region bottom_region block 0 ${Lmid} 0 ${Lywall} -0.01 0.01 units box' >> $dirname1/$input_create
               echo 'region mid_region block 0 ${Lmid} ${Lywall} ${walltopystart} -0.01 0.01 units box' >> $dirname1/$input_create
               echo 'region entire_region block 0 ${Lmid} 0 ${walltopyend} -0.01 0.01 units box' >> $dirname1/$input_create
               echo 'region shiftdownmid_region block 0 ${Lmid} 0 ${Lmid} -0.01 0.01 units box' >> $dirname1/$input_create
               
          # Defining the number of wall particles and total number of particles with the walls
               echo 'variable Nawall equal round(${phiwall}*${Lywall}*${Lmid}/(PI*(${Ra}^2+${Rb}^2)))' >> $dirname1/$input_create
               echo 'variable Nbwall equal ${Nawall}' >> $dirname1/$input_create
               echo 'variable Natot equal 2.0*${Nawall}+${Namid}' >> $dirname1/$input_create
               echo 'variable Nbtot equal 2.0*${Nbwall}+${Nbmid}' >> $dirname1/$input_create
               
               # Creating our region with the desired number of particles and pins
               echo 'create_box 3 entire_region' >> $dirname1/$input_create
               echo 'create_atoms 1 random ${Natot} '${random_num1[$index]}' entire_region' >> $dirname1/$input_create
               echo 'create_atoms 2 random ${Nbtot} '${random_num2[$index]}' entire_region' >> $dirname1/$input_create
               echo 'create_atoms 3 region shiftdownmid_region basis 1 3' >> $dirname1/$input_create
               
               # Defining mass of the particles
               echo 'mass 1 1.0' >> $dirname1/$input_create
               echo 'mass 2 1.0' >> $dirname1/$input_create
               echo 'mass 3 1.0' >> $dirname1/$input_create
               
               # Creating groups. Here, we separate by mobile particles and pins
               echo 'group particlesentire type 1 2' >> $dirname1/$input_create
               echo 'group pins type 3' >> $dirname1/$input_create
               echo 'displace_atoms pins move 0 ${Lywall} 0 units box' >> $dirname1/$input_create 
               # Setting velocity to zero and ensuring our simulation box is in 2-dimensions.
               echo 'velocity all set 0. 0. 0. units box' >> $dirname1/$input_create
               echo 'set group all vz 0.0 z 0.0' >> $dirname1/$input_create
               echo 'fix 2dFz0 all enforce2d' >> $dirname1/$input_create
               
               # Setting attraction to zero. Necessary inputs for modified dpd style from Irani. We do not want attraction.
               echo 'variable uIrani equal 0.0' >> $dirname1/$input_create
               echo 'variable oneplusu equal 1.0+${uIrani}' >> $dirname1/$input_create
               echo 'variable oneplus2u equal 1.0+2.0*${uIrani}' >> $dirname1/$input_create
               
               # Defining pair_style conditions
               echo 'pair_style  dpd/attract2 ${Tempr} ${Dmax} ${oneplus2u} ${oneplusu} 3297' >> $dirname1/$input_create
               echo 'pair_coeff  1 1 ${Aaa} ${gammadpdaa} ${Daa} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_create
               echo 'pair_coeff  1 2 ${Aab} ${gammadpdab} ${Dab} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_create
               echo 'pair_coeff  2 2 ${Abb} ${gammadpdbb} ${Dbb} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_create
               echo 'pair_coeff  1 3 ${Aac} ${gammadpdac} ${Dac} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_create
               echo 'pair_coeff  2 3 ${Abc} ${gammadpdbc} ${Dbc} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_create
               echo 'pair_coeff  3 3 ${Acc} ${gammadpdcc} ${Dcc} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_create

               # "Ghost" atoms can still store velocity information
               echo 'comm_modify vel yes' >> $dirname1/$input_create
               
               # Inputting timestep 
               echo 'timestep ${tstep}' >> $dirname1/$input_create
               
               # Neighborlist
               echo 'neighbor          0.3 bin' >> $dirname1/$input_create
               echo 'neigh_modify      every 1 delay 0 check yes' >> $dirname1/$input_create
          
               # Computing pos, velocity, and forces per atom
               echo 'compute unX all property/atom x y z vx vy vz fx fy fz' >> $dirname1/$input_create
               
               # This block of code allows us to track individual particles. 
               echo 'variable specpart string 2500' >> $dirname1/$input_create
               echo 'variable posxpart equal c_unX[${specpart}][1]' >> $dirname1/$input_create
               echo 'variable posypart equal c_unX[${specpart}][2]' >> $dirname1/$input_create
               echo 'variable poszpart equal c_unX[${specpart}][3]' >> $dirname1/$input_create
               echo 'variable vxpart equal c_unX[${specpart}][4]' >> $dirname1/$input_create
               echo 'variable vypart equal c_unX[${specpart}][5]' >> $dirname1/$input_create
               echo 'variable vzpart equal c_unX[${specpart}][6]' >> $dirname1/$input_create
               echo 'variable fxpart equal c_unX[${specpart}][7]' >> $dirname1/$input_create
               echo 'variable fypart equal c_unX[${specpart}][8]' >> $dirname1/$input_create
               echo 'variable fzpart equal c_unX[${specpart}][9]' >> $dirname1/$input_create
               
               # This is actual time. Our timestep times the MD step. 
               echo 'variable timehere equal ${tstep}*step' >> $dirname1/$input_create
               
               # Computing shear stress per particle 
               echo 'compute shearStr all stress/atom  thermo_temp virial' >> $dirname1/$input_create  #works only if thermo_temp and virial specified (different Eshan, Gaurav)  virial means kinetic term of stress tensor = 0
               echo 'variable stressxxi atom c_shearStr[1]' >> $dirname1/$input_create
               echo 'variable stressyyi atom c_shearStr[2]' >> $dirname1/$input_create

               echo 'variable Ntotentire equal ${Natot}+${Nbtot}+${Npin}' >> $dirname1/$input_create
               echo 'variable Areaentire equal ${Lmid}*${walltopyend}' >> $dirname1/$input_create
               
               # Computing pressure 
               echo 'variable piviaSxxSyyentire atom (-0.5)*(v_Ntotentire)/(v_Areaentire)*(v_stressxxi+v_stressyyi)' >> $dirname1/$input_create

               echo 'compute  vpressentire all pressure thermo_temp virial' >> $dirname1/$input_create
               echo 'variable vpressXYentire equal c_vpressentire[4]' >> $dirname1/$input_create # further pcheck see July 2021
               
          # Defining thermo_style and thermo frequency for the minimization part of simulation
               echo 'thermo_style custom step pe ke v_posxpart v_posypart v_vxpart v_vypart v_fxpart v_fypart  press  v_vpressXYentire v_timehere' >> $dirname1/$input_create   #does not have sigma_xy_entire (propor pXY)
               echo 'thermo_modify format line "%ld %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"' >> $dirname1/$input_create
               echo 'thermo 20000' >> $dirname1/$input_create
               
               # Defining dump for the minimization part of the simulation
               echo 'dump mydumpmin all custom 50000 Dump_Files/confdumpbeforemin.*.data id type x y  vx vy  fx fy xu yu c_shearStr[4] c_shearStr[1] c_shearStr[2] v_piviaSxxSyyentire' >> $dirname1/$input_create #during 1st min;prints at end of minimiz so freq > minstepmax
               echo 'dump_modify mydumpmin format line "%d %d %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e"' >> $dirname1/$input_create
               echo 'dump_modify mydumpmin sort id' >> $dirname1/$input_create
          #    following two lines need to be before minimization
               
          # Ensuring particles are fixed to lattice. 
               echo 'velocity pins set 0.0 0.0 0.0 units box' >> $dirname1/$input_create
               echo 'fix freeze pins setforce 0.0 0.0 0.0' >> $dirname1/$input_create #for minimization has to be this because for minimization nve/noforce does not keep velocities zero
               
          # Minimization 
               echo 'min_style fire' >> $dirname1/$input_create
               echo 'min_modify line quadratic' >> $dirname1/$input_create
               echo 'minimize 1e-200 1e-200 1000000 1000000' >> $dirname1/$input_create
               echo 'unfix freeze' >> $dirname1/$input_create
               echo 'undump mydumpmin' >> $dirname1/$input_create
               
               # Redefining box. Here we are changing boundary condition so that periodic images are only in x-direction. The set group all image commands ensures all particles are in simulation box after minimization and before we group particles into regions.
               echo 'set group all  image 0 0 0' >> $dirname1/$input_create
               echo 'change_box all boundary p f p' >> $dirname1/$input_create
               
               # Creating groups to separate each regions particles 
               echo 'group topwallparticles region  top_region' >> $dirname1/$input_create
               echo 'group bottomwallparticles region  bottom_region' >> $dirname1/$input_create
               echo 'group midparticles region  mid_region' >> $dirname1/$input_create
               echo 'group wallparticles union topwallparticles bottomwallparticles' >> $dirname1/$input_create
               echo 'group midmoveparticles subtract midparticles pins' >> $dirname1/$input_create
               
               # Setting wall particles velocity to zero. We are preparing to shear. 
               echo 'velocity wallparticles set 0.0 0.0 0.0 units box' >> $dirname1/$input_create
               
               # Resetting timestep and then using a write data in case we want to start the simulation after minimization. 
               echo 'reset_timestep 0' >> $dirname1/$input_create
               echo 'write_data readin_after1stmin_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote>> $dirname1/$input_create
               
               # Creating variable to know the total number of mid particles 
               echo 'variable Nmid equal count(midparticles,mid_region)' >> $dirname1/$input_create
               
               # Dump command for relaxation
               echo 'dump mydumpallMDnoshear all custom 50000 Dump_Files/confdumpallMDnoshear.*.data id type x y  vx vy  fx fy xu yu c_shearStr[4] c_shearStr[1] c_shearStr[2]' >> $dirname1/$input_create
               echo 'dump_modify mydumpallMDnoshear format line "%d %d %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e"' >> $dirname1/$input_create
               echo 'dump_modify mydumpallMDnoshear sort id' >> $dirname1/$input_create
          
               # Computing stress and pressure for the mid region during relaxation
               echo 'compute shearStrmid midparticles reduce ave c_shearStr[1] c_shearStr[2] c_shearStr[3] c_shearStr[4] c_shearStr[5] c_shearStr[6]' >> $dirname1/$input_create

          # variables def.for thermo (and could be used for printing into file info(t))
          # XX c_shearStrmid[1]   YY [2]   ZZ [3]    XY [4]   XZ [5]     YZ [6]
               echo 'variable shearXYmid equal c_shearStrmid[4]' >> $dirname1/$input_create

               echo 'compute shearStrmidmove midmoveparticles reduce ave c_shearStr[1] c_shearStr[2] c_shearStr[3] c_shearStr[4] c_shearStr[5] c_shearStr[6]' >> $dirname1/$input_create
               echo 'variable shearXYmidmove equal c_shearStrmidmove[4]' >> $dirname1/$input_create

               echo 'variable Nmidmove equal ${Nmid}-${Npin}' >> $dirname1/$input_create
               echo 'variable Areamid equal ${Lmid}*${Lmid}' >> $dirname1/$input_create

               echo 'variable pviaSxxSyymidmove equal (-1.0)*(${Nmidmove}/${Areamid})*0.5*(c_shearStrmidmove[1]+c_shearStrmidmove[2])' >> $dirname1/$input_create
               echo 'variable pviaSxxSyymid equal (-1.0)*(${Nmid}/${Areamid})*0.5*(c_shearStrmid[1]+c_shearStrmid[2])' >> $dirname1/$input_create
               
               # thermo style for relaxation steps
               echo 'thermo_style custom step pe ke v_posxpart v_posypart v_vxpart v_vypart v_fxpart v_fypart press v_pviaSxxSyymid v_pviaSxxSyymidmove v_shearXYmid v_shearXYmidmove' >> $dirname1/$input_create
               echo 'thermo_modify format line "%ld %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g  %20.15g %20.15g"' >> $dirname1/$input_create
               echo 'thermo 10000' >> $dirname1/$input_create
               
               # Ensuring wall particles are not moving. Preparing for shear. 
               echo 'velocity wallparticles set 0. 0. 0. units box' >> $dirname1/$input_create
               echo 'fix freezewallpart wallparticles nve/noforce' >> $dirname1/$input_create
     
               # Ensuring pins are fixed still after minimization and relaxation. 
               echo 'velocity pins set 0.0 0.0 0.0 units box' >> $dirname1/$input_create
               echo 'fix freezepins pins nve/noforce' >> $dirname1/$input_create
               
               # Ensuring particles velocity is zero and allowing only midparticles to move. (nve)
               echo 'velocity midmoveparticles set 0.0 0.0 0.0 units box' >> $dirname1/$input_create
               echo 'fix nveMDnoshear midmoveparticles nve' >> $dirname1/$input_create
               
               # Running relaxation region and writing a read data and restart files in case we want to start at this point of the simulation. 
               echo 'reset_timestep 0' >> $dirname1/$input_create
               echo 'run 500000' >> $dirname1/$input_create
               #echo 'write_restart Restart_Files/Restart_afterMDnoshear_Namid'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_create
               echo 'set group all  image 0 0 0' >> $dirname1/$input_create
               echo 'write_data readin_afterMDnoshear_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_create
               echo 'undump mydumpallMDnoshear' >> $dirname1/$input_create
               echo 'unfix nveMDnoshear' >> $dirname1/$input_create
               echo 'unfix freezewallpart' >> $dirname1/$input_create
               echo 'unfix freezepins' >> $dirname1/$input_create

          #------------ lines needed when MD with shear starts with read_data: --------
                    
                    input_read_str="input_Npin"$Pins"_Phi"$Phistring"_gammadot"$gammadotbeforepoint"eminus"$gammadote"_read"
                    output_read_str="output_Npin"$Pins"_Phi"$Phistring"_gammadot"$gammadotbeforepoint"eminus"$gammadote"_read"
                    
                    # read_data_func () {
                    #      echo 'read_data readin_afterMD'$1'_NA'$2'_Npin'$3'_Phi'$4'_gammadot'$5'eminus'$6 >> $dirname1/$input_read  
                    # }


                    # Defining the number of MD steps for each region	       
                    MD_elastic_strain_max=($(bc <<< "$elastic_strain_max/($DeltatMDshear*$gammadot)"))
                    MD_transient_strain_max=($(bc <<< "$transient_strain_max/($DeltatMDshear*$gammadot)"))
                    MD_eq_strain_max=($(bc <<< "$eq_strain_max/($DeltatMDshear*$gammadot)")) 
                    MD_inf_strain_max=($(bc <<< "$inf_strain_max/($DeltatMDshear*$gammadot)"))
                    MD_eq_strain_third=($(bc <<< "$MD_eq_strain_max/3"))              

                    # Defining the frequency at which we print dump files 
                    conf_elastic_MD=($(bc <<< "$dump_elastic_freq/($Deltatcreate*$gammadot)"))
                    conf_transient_MD=($(bc <<< "$dump_transient_freq/($Deltatcreate*$gammadot)"))
                    conf_eq_MD=($(bc <<< "$dump_eq_freq/($Deltatcreate*$gammadot)"))
                    conf_inf_MD=($(bc <<< "$dump_inf_freq/($Deltatcreate*$gammadot)")) #changed # bash calculator always produces floor integer; in order to avoid 0, scale provides decimal points
                    
                    # Defining the frequency at which we record thermo data 
                    thermo_elastic_MD=($(bc <<< "$thermo_elastic_freq/($Deltatcreate*$gammadot)"))
                    thermo_transient_MD=($(bc <<< "$thermo_transient_freq/($Deltatcreate*$gammadot)"))
                    thermo_eq_MD=($(bc <<< "$thermo_eq_freq/($Deltatcreate*$gammadot)"))
                    thermo_inf_MD=($(bc <<< "$thermo_inf_freq/($Deltatcreate*$gammadot)"))

                    # Defining the frequency at which we record chunk average
                    chunk_eq_MD=($(bc <<< "$chunk_eq_freq/($Deltatcreate*$gammadot)"))
                    chunk_inf_MD=($(bc <<< "$chunk_inf_freq/($Deltatcreate*$gammadot)"))
                    # Defining Nevery and Nrepeat in MD steps
                    Nevery_infMD=($(bc <<< "$Nevery_inf/($Deltatcreate*$gammadot)"))
                    Nevery_eqMD=($(bc <<< "$Nevery_eq/($Deltatcreate*$gammadot)"))
                    #Nrepeat_MD=($(bc <<< "$Nrepeat/($Deltatcreate*$gammadot)"))
                    
                    # for index in "${!regioname[@]}";
                    # do
                    input_read=$input_read_str
                    

                    # Same code as before, but now we are doing a read in after the create. 
                    echo ' '> $dirname1/$input_read
                    echo 'units lj' >> $dirname1/$input_read
                    echo 'dimension 2' >> $dirname1/$input_read
                    echo 'atom_style atomic' >> $dirname1/$input_read
                    echo 'atom_modify map array' >> $dirname1/$input_read
                    echo 'boundary p f p' >> $dirname1/$input_read  
                    echo 'variable tstep equal ' $DeltatMDshear >> $dirname1/$input_read
                    echo 'variable Tempr equal 0.0' >> $dirname1/$input_read

     # Same as before 

                    echo 'variable gammadoterate equal' $gammadot >> $dirname1/$input_read
                    echo 'variable gammadpd equal 1.0' >> $dirname1/$input_read
                    echo 'variable gammadpdaa equal ${gammadpd}' >> $dirname1/$input_read
                    echo 'variable gammadpdab equal ${gammadpd}' >> $dirname1/$input_read
                    echo 'variable gammadpdbb equal ${gammadpd}' >> $dirname1/$input_read
                    echo 'variable gammadpdac equal ${gammadpd}' >> $dirname1/$input_read
                    echo 'variable gammadpdbc equal ${gammadpd}' >> $dirname1/$input_read
                    echo 'variable gammadpdcc equal ${gammadpd}' >> $dirname1/$input_read
                    echo 'variable Namid equal' $Namid >> $dirname1/$input_read
                    echo 'variable Nbmid equal' $Nbmid >> $dirname1/$input_read
                    echo 'variable Ntotmid equal ${Namid}+${Nbmid}' >> $dirname1/$input_read
                    echo 'variable phi equal' $Phi >> $dirname1/$input_read
                    echo 'variable phiwall equal ${phi}' >> $dirname1/$input_read
                    echo 'variable Ra equal 1.0' >> $dirname1/$input_read
                    echo 'variable Rb equal 1.4' >> $dirname1/$input_read
                    echo 'variable Rpin equal ${Ra}*(0.004)' >> $dirname1/$input_read
                    echo 'variable eps equal 1.0' >> $dirname1/$input_read
                    echo 'variable Daa equal ${Ra}+${Ra}' >> $dirname1/$input_read
                    echo 'variable Dab equal ${Ra}+${Rb}' >> $dirname1/$input_read
                    echo 'variable Dbb equal ${Rb}+${Rb}' >> $dirname1/$input_read
                    echo 'variable Dac equal ${Ra}+${Rpin}' >> $dirname1/$input_read
                    echo 'variable Dbc equal ${Rb}+${Rpin}' >> $dirname1/$input_read
                    echo 'variable Dcc equal ${Rpin}+${Rpin}' >> $dirname1/$input_read
                    echo 'variable Aaa equal ${eps}/${Daa}' >> $dirname1/$input_read
                    echo 'variable Aab equal ${eps}/${Dab}' >> $dirname1/$input_read
                    echo 'variable Abb equal ${eps}/${Dbb}' >> $dirname1/$input_read
                    echo 'variable Aac equal ${eps}/${Dac}' >> $dirname1/$input_read
                    echo 'variable Abc equal ${eps}/${Dbc}' >> $dirname1/$input_read
                    echo 'variable Acc equal ${eps}/${Dcc}' >> $dirname1/$input_read
                    echo 'variable Dmax equal ${Dbb}' >> $dirname1/$input_read
                    echo 'variable Lmid equal sqrt(PI*((${Ra}^2)*${Namid}+(${Rb}^2)*${Nbmid})/${phi})' >> $dirname1/$input_read
                    echo 'variable Npin equal' $Pins >> $dirname1/$input_read
                    echo 'variable Nbasis equal 1' >> $dirname1/$input_read
                    echo 'variable Npinx equal sqrt(${Npin})' >> $dirname1/$input_read
                    echo 'variable Lywall equal  ${Dbb}*3.0' >> $dirname1/$input_read
                    echo 'variable walltopystart equal ${Lywall}+${Lmid}' >> $dirname1/$input_read
                    echo 'variable Lchunk equal (${walltopystart}-${Lywall})/32' >> $dirname1/$input_read
                    echo 'variable walltopyend equal 2.0*${Lywall}+${Lmid}' >> $dirname1/$input_read
                    echo 'variable Lunitcellx equal ${Lmid}/${Npinx}' >> $dirname1/$input_read
                    echo 'variable Lunitcelly equal ${walltopystart}/${Npinx}' >> $dirname1/$input_read
                    echo 'variable LunitcellLywall equal ${Lywall}/${Npinx}' >> $dirname1/$input_read
                    echo 'variable Lunitcelltopyend equal ${walltopyend}/${Npinx}' >> $dirname1/$input_read

               
                    echo 'region top_region block 0 ${Lmid} ${walltopystart} ${walltopyend} -0.01 0.01 units box' >> $dirname1/$input_read
                    echo 'region bottom_region block 0 ${Lmid} 0 ${Lywall} -0.01 0.01 units box' >> $dirname1/$input_read
                    echo 'region mid_region block 0 ${Lmid} ${Lywall} ${walltopystart} -0.01 0.01 units box' >> $dirname1/$input_read
                    echo 'region entire_region block 0 ${Lmid} 0 ${walltopyend} -0.01 0.01 units box' >> $dirname1/$input_read
                    echo 'variable Nawall equal round(${phiwall}*${Lywall}*${Lmid}/(PI*(${Ra}^2+${Rb}^2)))' >> $dirname1/$input_read
                    echo 'variable Nbwall equal ${Nawall}' >> $dirname1/$input_read
                    echo 'variable Natot equal 2.0*${Nawall}+${Namid}' >> $dirname1/$input_read
                    echo 'variable Nbtot equal 2.0*${Nbwall}+${Nbmid}' >> $dirname1/$input_read
               
                    echo 'variable uIrani equal 0.0' >> $dirname1/$input_read
                    echo 'variable oneplusu equal 1.0+${uIrani}' >> $dirname1/$input_read
                    echo 'variable oneplus2u equal 1.0+2.0*${uIrani}' >> $dirname1/$input_read

                    echo 'pair_style  dpd/attract2 ${Tempr} ${Dmax} ${oneplus2u} ${oneplusu} 3297' >> $dirname1/$input_read
                    echo 'read_data readin_afterMDnoshear_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_read  
                    # Reading in data after relaxation stage 
                    # if [ $index -eq 0 ]
                    # then
                    #      read_data_func "noshear" $Namid $Pins $Phistring $gammadotbeforepoint $gammadote
                    # elif [ $index -eq 1 ]
                    # then
                    #      read_data_func "transient" $Namid $Pins $Phistring $gammadotbeforepoint $gammadote
                    # elif [ $index -eq 2 ]
                    # then
                    #      read_data_func "equil1" $Namid $Pins $Phistring $gammadotbeforepoint $gammadote
                    # elif [ $index -eq 3 ]
                    # then
                    #      read_data_func "equil2" $Namid $Pins $Phistring $gammadotbeforepoint $gammadote
                    # else
                    #      read_data_func "equil3" $Namid $Pins $Phistring $gammadotbeforepoint $gammadote
                    # fi

                    echo 'pair_coeff  1 1 ${Aaa} ${gammadpdaa} ${Daa} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_read
                    echo 'pair_coeff  1 2 ${Aab} ${gammadpdab} ${Dab} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_read
                    echo 'pair_coeff  2 2 ${Abb} ${gammadpdbb} ${Dbb} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_read
                    echo 'pair_coeff  1 3 ${Aac} ${gammadpdac} ${Dac} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_read
                    echo 'pair_coeff  2 3 ${Abc} ${gammadpdbc} ${Dbc} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_read
                    echo 'pair_coeff  3 3 ${Acc} ${gammadpdcc} ${Dcc} ${oneplus2u} ${oneplusu}' >> $dirname1/$input_read

                    echo 'group particlesentire type 1 2' >> $dirname1/$input_read
                    echo 'group pins type 3' >> $dirname1/$input_read

                    echo 'velocity all set 0. 0. 0. units box' >> $dirname1/$input_read
                    echo 'set group all vz 0.0 z 0.0' >> $dirname1/$input_read
                    echo 'fix 2dFz0 all enforce2d' >> $dirname1/$input_read
                    echo 'comm_modify vel yes' >> $dirname1/$input_read
                    echo 'timestep ${tstep}' >> $dirname1/$input_read
                    echo 'neighbor          0.3 bin' >> $dirname1/$input_read
                    echo 'neigh_modify      every 1 delay 0 check yes' >> $dirname1/$input_read
                    echo 'compute unX all property/atom x y z vx vy vz fx fy fz' >> $dirname1/$input_read
                    echo 'variable specpart string 101' >> $dirname1/$input_read
                    echo 'variable posxpart equal c_unX[${specpart}][1]' >> $dirname1/$input_read
                    echo 'variable posypart equal c_unX[${specpart}][2]' >> $dirname1/$input_read
                    echo 'variable poszpart equal c_unX[${specpart}][3]' >> $dirname1/$input_read
                    echo 'variable vxpart equal c_unX[${specpart}][4]' >> $dirname1/$input_read
                    echo 'variable vypart equal c_unX[${specpart}][5]' >> $dirname1/$input_read
                    echo 'variable vzpart equal c_unX[${specpart}][6]' >> $dirname1/$input_read
                    echo 'variable fxpart equal c_unX[${specpart}][7]' >> $dirname1/$input_read
                    echo 'variable fypart equal c_unX[${specpart}][8]' >> $dirname1/$input_read
                    echo 'variable fzpart equal c_unX[${specpart}][9]' >> $dirname1/$input_read

                    echo 'variable timehere equal ${tstep}*step' >> $dirname1/$input_read

                    echo 'group topwallparticles region  top_region' >> $dirname1/$input_read
                    echo 'group bottomwallparticles region  bottom_region' >> $dirname1/$input_read
                    echo 'group midparticles region  mid_region' >> $dirname1/$input_read
                    echo 'group wallparticles union topwallparticles bottomwallparticles' >> $dirname1/$input_read
                    echo 'group midmoveparticles subtract midparticles pins' >> $dirname1/$input_read
                    echo 'group particleA type 1' >> $dirname1/$input_read
                    echo 'group particleB type 2' >> $dirname1/$input_read
                    echo 'group type1mid intersect midparticles particleA' >> $dirname1/$input_read   #changed # grouping only type 1 midparticles
                    echo 'group type2mid intersect midparticles particleB' >> $dirname1/$input_read    #changed # grouping only type 2 midparticles 
                    echo 'group type1wall intersect wallparticles particleA' >> $dirname1/$input_read  #changed # grouping only type 1 wallparticles
                    echo 'group type2wall intersect wallparticles particleB' >> $dirname1/$input_read   #changed # grouping only type 2 wallparticles
               
                    echo 'compute shearStr all  stress/atom  thermo_temp virial' >> $dirname1/$input_read  #works only if thermo_temp and virial specified (different Eshan, Gaurav)  virial means kinetic term of stress tensor = 0
               #-----
                    echo 'variable Nmid equal count(midparticles,mid_region)' >> $dirname1/$input_read

                    echo 'compute shearStrmid midparticles reduce ave c_shearStr[1] c_shearStr[2] c_shearStr[3] c_shearStr[4] c_shearStr[5] c_shearStr[6]' >> $dirname1/$input_read

               # variables def.for thermo (and could be used for printing into file info(t))
               # XX c_shearStrmid[1]   YY [2]   ZZ [3]    XY [4]   XZ [5]     YZ [6]
                    echo 'variable shearXYmid equal c_shearStrmid[4]' >> $dirname1/$input_read

                    echo 'compute shearStrmidmove midmoveparticles reduce ave c_shearStr[1] c_shearStr[2] c_shearStr[3] c_shearStr[4] c_shearStr[5] c_shearStr[6]' >> $dirname1/$input_read
                    echo 'variable shearXYmidmove equal c_shearStrmidmove[4]' >> $dirname1/$input_read
                    echo 'compute shearstressMDwall wallparticles stress/atom thermo_temp virial' >> $dirname1/$input_read
                    echo 'variable Nmidmove equal ${Nmid}-${Npin}' >> $dirname1/$input_read
                    echo 'variable Areamid equal ${Lmid}*${Lmid}' >> $dirname1/$input_read

                    echo 'variable pviaSxxSyymidmove equal (-1.0)*(${Nmidmove}/${Areamid})*0.5*(c_shearStrmidmove[1]+c_shearStrmidmove[2])' >> $dirname1/$input_read
                    echo 'variable pviaSxxSyymid equal (-1.0)*(${Nmid}/${Areamid})*0.5*(c_shearStrmid[1]+c_shearStrmid[2])' >> $dirname1/$input_read
               #------- end of lines needed when MD with shear starts with read_data --------
               # MD run with shear:
                    echo 'velocity midparticles set 0. 0. 0. units box' >> $dirname1/$input_read
                    echo 'variable vtop equal 0.5*${Lmid}*${gammadoterate}' >> $dirname1/$input_read
                    echo 'variable vbottom equal (-1.0)*0.5*${Lmid}*${gammadoterate}' >> $dirname1/$input_read
                    echo 'velocity topwallparticles set ${vtop} 0. 0. units box' >> $dirname1/$input_read
                    echo 'velocity bottomwallparticles set ${vbottom} 0. 0. units box' >> $dirname1/$input_read
                    echo 'fix freezetop topwallparticles nve/noforce' >> $dirname1/$input_read
                    echo 'fix freezebottom bottomwallparticles nve/noforce' >> $dirname1/$input_read
                    echo 'velocity pins set 0.0 0.0 0.0 units box' >> $dirname1/$input_read
                    echo 'fix freezepins pins nve/noforce' >> $dirname1/$input_read

                    echo 'fix nveMD midmoveparticles nve' >> $dirname1/$input_read
                    
                    # Defining a variable for the strain 
                    echo 'variable strain equal  ${tstep}*step*${gammadoterate}' >> $dirname1/$input_read
                    
                    # This section of code is to compute the coordination number of midparticles with only the wall   
                    echo 'compute 1_wall1 type1mid coord/atom cutoff ${Daa} group type1wall' >> $dirname1/$input_read
                    echo 'compute 1_wall2 type1mid coord/atom cutoff ${Dab} group type2wall' >> $dirname1/$input_read
                    echo 'compute 2_wall2 type2mid coord/atom cutoff ${Dbb} group type2wall' >> $dirname1/$input_read
                    echo 'compute 2_wall1 type2mid coord/atom cutoff ${Dab} group type1wall' >> $dirname1/$input_read
                    echo 'variable coordA_wall atom c_1_wall1+c_1_wall2' >> $dirname1/$input_read
                    echo 'variable coordB_wall atom c_2_wall2+c_2_wall1' >> $dirname1/$input_read

               # This section of code is to compute the coordination number of mobile particles   
                    echo 'compute 1_1 type1mid coord/atom cutoff ${Daa} group type1mid' >> $dirname1/$input_read
                    echo 'compute 1_2 type1mid coord/atom cutoff ${Dab} group type2mid' >> $dirname1/$input_read
                    echo 'compute 2_1 type2mid coord/atom cutoff ${Dbb} group type2mid' >> $dirname1/$input_read
                    echo 'compute 2_2 type2mid coord/atom cutoff ${Dab} group type1mid' >> $dirname1/$input_read
                    echo 'variable coordA atom c_1_1+c_1_2' >> $dirname1/$input_read
                    echo 'variable coordB atom c_2_1+c_2_2' >> $dirname1/$input_read

                    # This section of code is to compute the coordination number of mobile partciles only with pins 
                    echo 'compute 1_pin type1mid coord/atom cutoff ${Dac} group pins' >> $dirname1/$input_read
                    echo 'compute 2_pin type2mid coord/atom cutoff ${Dbc} group pins' >> $dirname1/$input_read
                    echo 'variable coordA_pin atom c_1_pin' >> $dirname1/$input_read
                    echo 'variable coordB_pin atom c_2_pin' >> $dirname1/$input_read

                    # chop MD run with shear into different regions:
               
                    # for elastic region:
                    
                    echo 'thermo_style custom step pe ke v_posxpart v_posypart v_vxpart v_vypart v_fxpart v_fypart press v_pviaSxxSyymid v_pviaSxxSyymidmove v_shearXYmid v_shearXYmidmove v_strain' >> $dirname1/$input_read
                    echo 'thermo_modify format line "%ld %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"' >> $dirname1/$input_read
                    echo 'thermo '$thermo_elastic_MD >> $dirname1/$input_read          

                    echo 'dump mydumpallMD all custom '$conf_elastic_MD' Dump_Files/confdumpallelasticMD*.data id type x y vx vy fx fy xu yu c_shearStr[4] c_shearStr[1] c_shearStr[2] v_coordA v_coordB v_coordA_wall v_coordB_wall v_coordA_pin v_coordB_pin' >> $dirname1/$input_read #changed (adding coord variables)
                    echo 'dump_modify mydumpallMD format line "%d %d %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %d %d %d %d %d %d"' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD sort id' >> $dirname1/$input_read
                    
                    # echo 'dump mydumpwall wallparticles custom '$conf_elastic_MD' Dump_Files/confdumpwallelasticMD.*.data id type x y  vx vy  fx fy xu yu c_shearstressMDwall[4]' >> $dirname1/$input_read
                    # echo 'dump_modify mydumpwall sort id' >> $dirname1/$input_read
                    
                    echo 'reset_timestep 0' >> $dirname1/$input_read
                    echo 'run '$MD_elastic_strain_max >> $dirname1/$input_read   #elastic (small t) of MD
                    echo 'write_data readin_afterMDelastic_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_read
                    
               #    transient: 
                         
                    echo 'thermo '$thermo_transient_MD >> $dirname1/$input_read 
                    
                    echo 'undump mydumpallMD' >> $dirname1/$input_read
                    echo 'dump mydumpallMD all custom '$conf_transient_MD' Dump_Files/confdumpalltransMD*.data id type x y  vx vy  fx fy xu yu c_shearStr[4] c_shearStr[1] c_shearStr[2] v_coordA v_coordB v_coordA_wall v_coordB_wall v_coordA_pin v_coordB_pin' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD format line "%d %d %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %d %d %d %d %d %d"' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD sort id' >> $dirname1/$input_read # sorting particles in the dump file
                    
                    # echo 'undump mydumpwall' >> $dirname1/$input_read
                    # echo 'dump mydumpwall wallparticles custom '$conf_transient_MD' Dump_Files/confdumpwalltransMD.*.data id type x y  vx vy  fx fy xu yu c_shearstressMDwall[4]' >> $dirname1/$input_read
                    # echo 'dump_modify mydumpwall sort id' >> $dirname1/$input_read
                    echo 'run '$MD_transient_strain_max >> $dirname1/$input_read  #transient 990000=1000000-10000
                    echo 'write_data readin_afterMDtransient_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_read
               
                    #    equilibration: 
                         
                    echo 'thermo '$thermo_eq_MD >> $dirname1/$input_read
                    echo 'undump mydumpallMD' >> $dirname1/$input_read
                    echo 'dump mydumpallMD all custom '$conf_eq_MD' Dump_Files/confdumpalleqMD*.data id type x y  vx vy  fx fy xu yu c_shearStr[4] c_shearStr[1] c_shearStr[2] v_coordA v_coordB v_coordA_wall v_coordB_wall v_coordA_pin v_coordB_pin' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD format line "%d %d %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %d %d %d %d %d %d"' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD sort id' >> $dirname1/$input_read
                    
                    # echo 'dump mydumpwall wallparticles custom '$conf_eq_MD' Dump_Files/confdumpwalleqMD.*.data id type x y  vx vy  fx fy xu yu c_shearstressMDwall[4]' >> $dirname1/$input_read
                    # echo 'dump_modify mydumpwall sort id' >> $dirname1/$input_read
                    
                    # chunking for velocity profile



                    echo 'compute chunkfix midmoveparticles chunk/atom bin/2d x 0.00 ${Lmid} y ${Lywall} ${Lchunk} bound y ${Lywall} ${walltopystart} units box discard mixed' >> $dirname1/$input_read
                    #Amy revision ... rename profilefix to profilefix_XXX so can do eq or inf or ... 
                    echo 'fix profilefix_eq midmoveparticles ave/chunk '$Nevery_eqMD' '$Nrepeat' '$chunk_eq_MD' chunkfix vx vy c_shearStr[4] c_shearStr[1] c_shearStr[2] norm sample ave one file profileq' >> $dirname1/$input_read

                    echo 'run '$MD_eq_strain_max >> $dirname1/$input_read   
                    echo 'write_data readin_afterMDequil_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_read

                    
                    # after equilibration time series: 
                    #echo 'reset_timestep 0' >> $dirname1/$input_read
                    echo 'thermo '$thermo_inf_MD >> $dirname1/$input_read                
                    echo 'undump mydumpallMD' >> $dirname1/$input_read
                    echo 'dump mydumpallMD all custom '$conf_inf_MD' Dump_Files/confdumpallinfMD*.data id type x y  vx vy  fx fy xu yu c_shearStr[4] c_shearStr[1] c_shearStr[2] v_coordA v_coordB v_coordA_wall v_coordB_wall v_coordA_pin v_coordB_pin' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD format line "%d %d %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %20.10e %d %d %d %d %d %d"' >> $dirname1/$input_read
                    echo 'dump_modify mydumpallMD sort id' >> $dirname1/$input_read
               
                    # echo 'dump mydumpwall wallparticles custom '$conf_inf_MD' Dump_Files/confdumpwallinfMD.*.data id type x y  vx vy  fx fy xu yu c_shearstressMDwall[4]' >> $dirname1/$input_read
                    # echo 'dump_modify mydumpwall sort id' >> $dirname1/$input_read
                    
                    # chunking for velocity profile

                    echo 'compute chunkfix_inf midmoveparticles chunk/atom bin/2d x 0.00 ${Lmid} y ${Lywall} ${Lchunk} bound y ${Lywall} ${walltopystart} units box discard mixed' >> $dirname1/$input_read
                    #Amy revision ... fix new profile_XXX for inf region ... but leave unfix for luck :-)
                    echo 'unfix profilefix_eq' >> $dirname1/$input_read
                    echo 'fix profilefix_inf midmoveparticles ave/chunk '$Nevery_infMD' '$Nrepeat' '$chunk_inf_MD' chunkfix_inf vx vy c_shearStr[4] c_shearStr[1] c_shearStr[2] norm sample ave one file profileinf' >> $dirname1/$input_read

                    echo 'run '$MD_inf_strain_max >> $dirname1/$input_read   #6e6-1e6=5e6

               
                    echo 'write_data readin_MDinf_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_read
                    echo 'write_restart Restart_Files/Restart_NA'$Namid'_Npin'$Pins'_Phi'$Phistring'_gammadot'$gammadotbeforepoint'eminus'$gammadote >> $dirname1/$input_read
                    echo 'write_dump wallparticles custom Dump_Files/wallpartidtype id type' >> $dirname1/$input_read #changed # creating a single file with wall particles
               
               
               
                    
                    #---------------------------------------o 
                         # Creating variables to store data into outfiles given a specific naming convention. 
                    output_create="output_Npin"$Pins"_Phi"$Phistring"_gammadot"$gammadotbeforepoint"eminus"$gammadote"_create"
                    output_read=$output_read_str
                    Make_run='Make_run.sh'                    
                    # Creating a run script named Make_run.sh. Our run script then runs the Make_run file. 
                    echo '#!/bin/bash' >> $dirname1/$Make_run
                    
                    echo '#SBATCH -p skx-normal # partition (queue)' >> $dirname1/$Make_run
                    
                    echo '#SBATCH -n 32 # number of cores' >> $dirname1/$Make_run  #changed # increasging the core number to run faster; 68 is the highest you can give for normal mode
                    echo '#SBATCH --job-name="june26_1e-6" # job name' >> $dirname1/$Make_run
                    echo '#SBATCH -o slurm.%N.%j.out # STDOUT' >> $dirname1/$Make_run
                    echo '#SBATCH -e slurm.%N.%j.err # STDERR' >> $dirname1/$Make_run
               
                    echo '#SBATCH -t 02:30:00' >> $dirname1/$Make_run
                    
                    
                    echo '#SBATCH -N 1 # number of nodes' >> $dirname1/$Make_run #changed # increasing the number of nodes to run faster, highest node number for normal mode is 256; however the credit cost is 0.8 SU per node per hour
                    
                    echo 'module load intel/18.0.2' >> $dirname1/$Make_run
                    echo 'module load impi/18.0.2' >> $dirname1/$Make_run
                    echo 'module load lammps/9Jan20' >> $dirname1/$Make_run

                    echo 'export OMP_NUM_THREADS=1' >> $dirname1/$Make_run
                    
                    echo 'ibrun lmp_Eshan_stampede -in '$input_create' > '$output_create >> $dirname1/$Make_run
                    echo 'ibrun lmp_Eshan_stampede -in '$input_read' > '$output_read >> $dirname1/$Make_run
          
                         
                    done     
               done
          done
     done
done
