{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1cfabd04",
   "metadata": {
    "id": "1cfabd04"
   },
   "source": [
    "For calculating d2min and saving it to a file, go to the block next to\n",
    "# Calculating d2min \n",
    "\n",
    "\n",
    "## The function path_name\n",
    "This function assumes your dump files or porfile files are in directories with specific structure. You can create this specific structure using the block next to the function dirpath.\n",
    "\n",
    "# -------------------------------------------------------\n",
    "change June 19, 2023:  match t (m+1 conf; next) and t-Delta t (m conf; xr_ref) \n",
    "\n",
    "change June 20, 2023: add deformation (Delta r)^2 = D2min_i(K=0) (potentially normalization factor)\n",
    "\n",
    "change June 21, 2023: (Delta r)^2 = D2min_i(K=1=unitaryMatrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f62453ea",
   "metadata": {
    "id": "f62453ea"
   },
   "source": [
    "# Importing essential modules "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "38ed0684-e10e-438c-8868-1f7a645fa83b",
   "metadata": {
    "id": "38ed0684-e10e-438c-8868-1f7a645fa83b"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.animation\n",
    "from matplotlib.animation import FuncAnimation\n",
    "from matplotlib.animation import FFMpegWriter\n",
    "import os\n",
    "import pickle\n",
    "import csv\n",
    "import statistics as stats\n",
    "%matplotlib notebook\n",
    "import numpy as np\n",
    "from scipy import optimize\n",
    "import urllib\n",
    "import math"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59f24af2",
   "metadata": {
    "id": "59f24af2"
   },
   "source": [
    "# Setting styles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2f9469d7",
   "metadata": {
    "id": "2f9469d7"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/wv/mlgf5z4x4p95cl12gb8k58_h0000gn/T/ipykernel_25945/2896151872.py:1: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.\n",
      "  mpl.style.use('seaborn-talk')\n"
     ]
    }
   ],
   "source": [
    "mpl.style.use('seaborn-talk') \n",
    "plt.rc('figure', figsize = (6, 5)) # Reduces overall size of figures\n",
    "plt.rc('axes', labelsize=16, titlesize=14)\n",
    "plt.rc('figure', autolayout = True) # Adjusts supblot parameters for new size\n",
    "plt.rcParams['text.usetex'] = True"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "66549151",
   "metadata": {
    "id": "66549151"
   },
   "source": [
    "# Defining functions for calculating D2min"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f4bddee3-3e52-4cef-9963-7e6af4acf6b1",
   "metadata": {
    "id": "f4bddee3-3e52-4cef-9963-7e6af4acf6b1"
   },
   "outputs": [],
   "source": [
    "# you can modify path_name function. The function currently returns directory address\n",
    "#of dump files on my local computer. \n",
    "\n",
    "def path_name(gammadot, pin_num, phi):\n",
    "    '''\n",
    "    returns full path to dump files.\n",
    "    Parameters:\n",
    "        - cwd (current working directory); str \n",
    "        - gammadot; str\n",
    "        - pin_num; int\n",
    "        - phi; float\n",
    "        - filename; str\n",
    "    returns path; str\n",
    "            \n",
    "    '''\n",
    "    cwd = os.getcwd()                          #getting current directory\n",
    "    #dir_name =  str(gammadot) + '/'            #example: 1e-06/\n",
    "    #dir_name += 'Pins' + str(pin_num) + '/'    #example: 1e-06/Pins9\n",
    "    dir_name = 'Pins' + str(pin_num) + '/'    #example: 1e-06/Pins9\n",
    "    dir_name += 'Phi0' + str(int(phi*1e5)) + '/'  #example: 1e-06/Pins9/Phi084500   \n",
    "    dir_name += 'Dump_Files/'                  #example: 1e-06/Pins9/Phi084500/Dump_Files/\n",
    "            \n",
    "    path = os.path.join(cwd, dir_name)\n",
    "    #example: [current directory]/1e-06/Pins9/Phi084500/Dump_Files/\n",
    "    \n",
    "    return path\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "615e98fe",
   "metadata": {
    "id": "615e98fe"
   },
   "source": [
    "\n",
    "# you can modify the next 5 lines. And the line with the variable rel_profile_dir.\n",
    "cwd = os.getcwd() \n",
    "gammadotbeforepoint_list = [1] \n",
    "gammadot_list = [6] \n",
    "pin_list = [64] \n",
    "phi_list = [0.835]\n",
    "\n",
    "# Creating a structure of directories and subdirectories, for example, \n",
    "# [current directory]/1e-06/Pins9/Phi084500/\n",
    "\n",
    "for i in range(len(gammadotbeforepoint_list)): \n",
    "    for j in range(len(gammadot_list)):\n",
    "    \n",
    "        # the name of the directory that is going to be created in that current directory. This new directory will\n",
    "        # contain different directories named as differnt shear rates. In the given structure, I kept the string\n",
    "        # empty. This means, all the directories named as different shear rates will be located in the current \n",
    "        # directory, not under a directory within the current directory.\n",
    "        \n",
    "        rel_profile_dir = '' # you can write any string here.\n",
    "        gammadot = str(gammadotbeforepoint_list[i]*10**((-1)*gammadot_list[j]))\n",
    "        dir_profile = os.path.join(cwd, rel_profile_dir)\n",
    "        newdir_gammadot = os.path.join(dir_profile, gammadot) # example: [current directory]/1e-06\n",
    "        # creating the new directory. exist_ok=True leaves directory unaltered. \n",
    "        os.makedirs(newdir_gammadot, exist_ok=True)\n",
    "\n",
    "        for k in range(len(pin_list)):\n",
    "\n",
    "            pin_dirname = 'Pins' + str(pin_list[k])\n",
    "            newdir_pin = os.path.join(newdir_gammadot, pin_dirname) # example: [current directory]/1e-06/Pins9\n",
    "            os.makedirs(newdir_pin, exist_ok=True)\n",
    "\n",
    "            for l in range(len(phi_list)):\n",
    "\n",
    "                phi_dirname = 'Phi0' + str(int(phi_list[l]*1e5))\n",
    "                newdir_phi = os.path.join(newdir_pin, phi_dirname) # example: [current directory]/1e-06/Pins9/Phi084500\n",
    "                os.makedirs(newdir_phi, exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "62b76089-af3f-4f31-a832-61bbe350552a",
   "metadata": {
    "id": "62b76089-af3f-4f31-a832-61bbe350552a"
   },
   "outputs": [],
   "source": [
    "def system_info(path, ref_file):\n",
    "    '''returns total particle number (Ntot) and length of box in x-direction from the reference file\n",
    "    Unless total number of particles and length of box is changing over time, you should only use this function once.\n",
    "    Parameters:\n",
    "    path; str; the address of the directory containing dump files\n",
    "    ref_file; str; reference confdump file to get Ntot, Lx\n",
    "    '''\n",
    "    ref_address = path + ref_file \n",
    "    open_file = open(ref_address, 'r')\n",
    "    countline = 0\n",
    "\n",
    "    # Reading in reference file to determine Ntot, Lx for the configuration\n",
    "    \n",
    "    for line in open_file.readlines(): # goes line by line in the confdump file\n",
    "        countline +=1\n",
    "        fields = line.split() \n",
    "        # as line is a string, fields is a list of strings separated by any space in line\n",
    "\n",
    "        if countline == 4: #4th line of the confdump file\n",
    "            Ntot = int(fields[0]) # 1st (only) element of 4th line\n",
    "            \n",
    "        elif countline == 6: # 6th line\n",
    "            Lx = float(fields[1]) # 2nd (last) element of 6th line\n",
    "            \n",
    "            break # file-reading is done after line 6\n",
    "\n",
    "    open_file.close()\n",
    "    \n",
    "    return Ntot, Lx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9134dc4a-9fc4-4c22-a272-11506cf7d1d4",
   "metadata": {
    "id": "9134dc4a-9fc4-4c22-a272-11506cf7d1d4"
   },
   "outputs": [],
   "source": [
    "def id_pos_list(Ntot, MD_step, path, file):\n",
    "    '''extracts xy position and ID of each particle from a dump file\n",
    "    Parameters:\n",
    "    - Ntot; int; number of total particles \n",
    "    - MD_step; int; the MDstep number of the confdump file\n",
    "    - path; str; full path of the directory containing dump files \n",
    "    - file; str; name of the confdump file\n",
    "    Returns\n",
    "    xy_pairs; dictionary; key: (int) particle ID and value: tuple (x,y)\n",
    "    '''\n",
    "    \n",
    "    file = path + file\n",
    "    \n",
    "    xy_pairs = {} # initiating dictionary \n",
    "    \n",
    "    with open(file) as csvfile:\n",
    "        \n",
    "        csvreader = csv.reader(csvfile, delimiter=' ') \n",
    "        for n in range(9): # skipping first 9 rows in the dump file\n",
    "            next(csvreader)\n",
    "            \n",
    "        for row in csvreader: # row is a list of contents (string) in a row\n",
    "        \n",
    "            row = [x for x in row if x != '']\n",
    "\n",
    "            xy_pairs[int(row[0])] = ( float(row[2]), float(row[3]) )\n",
    "            # row[0]=particle ID, row[2]=x-position of the particle, row[3]=y-position\n",
    "    \n",
    "    return xy_pairs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "02bf6481-e23a-4053-b221-dc9033265d15",
   "metadata": {
    "id": "02bf6481-e23a-4053-b221-dc9033265d15"
   },
   "outputs": [],
   "source": [
    "def neighbor(Lx, Ntot, xy_pairs):\n",
    "    '''Saves neighbor information about each particle\n",
    "    Parameters:\n",
    "    Lx; int; x-length of system\n",
    "    Ntot; int; total number of particles\n",
    "    xy_pairs; dictionary; returned id_pos_list\n",
    "    Returns:\n",
    "    neigh_dict; dictionary; \n",
    "                key: particle ID (int), value: (sorted) lists of neighbor particle IDs.\n",
    "    r_dict; dictionary; \n",
    "                key: particle ID (int); value: lists of tuples (x,y) of neighbor particles.\n",
    "                                                Tuples sorted by neighbor IDs. \n",
    "    '''\n",
    "    \n",
    "    neigh_dict = {}\n",
    "    r_dict = {}\n",
    "    \n",
    "    for i in range(1,Ntot+1):\n",
    "        # ininitating empty list for each particle ID (key) in the dictionaries\n",
    "        neigh_dict[i] = []\n",
    "        r_dict[i] = []\n",
    "        \n",
    "    for i in range(1,Ntot+1):\n",
    "        # getting particle position (x,y)\n",
    "        xi = xy_pairs[i][0]\n",
    "        yi = xy_pairs[i][1]\n",
    "      \n",
    "        for j in range(i+1, Ntot+1): \n",
    "            # we have already updated neighbors for IDs less than i in previous iterations\n",
    "            # go through the full codes below; you will notice \n",
    "            \n",
    "            # getting potential neighbor position (x,y)\n",
    "            \n",
    "            \n",
    "            xj = xy_pairs[j][0]\n",
    "            \n",
    "            yj = xy_pairs[j][1]\n",
    "            # getting the distance between a particle and a potential neighbor in both directions\n",
    "            xr = xj-xi\n",
    "            yr = yj-yi\n",
    "            \n",
    "            # periodic boundary conditions\n",
    "            if xr > Lx/2: xr -= Lx \n",
    "            elif xr < - Lx/2: xr += Lx\n",
    "                \n",
    "            # distance squared\n",
    "            rijto2 = xr * xr + yr * yr\n",
    "            \n",
    "            if rijto2 < 36: # distance < 6\n",
    "                neigh_dict[i].append(j) # appending j to neighbor list of i\n",
    "                neigh_dict[j].append(i) \n",
    "                # as i is also a neighbor to j; also this is updating the neighbors of j, who are less than j\n",
    "                r_dict[i].append( (xr, yr) ) # appending j's distance from i \n",
    "                r_dict[j].append( (-xr, -yr) ) \n",
    "                # appending i's distance from j; updating the distances of neighbors of j, who are less than j\n",
    "        \n",
    "    return neigh_dict, r_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "dfbd5df3-a006-47e1-8ff3-d5f130f0eb36",
   "metadata": {
    "id": "dfbd5df3-a006-47e1-8ff3-d5f130f0eb36",
    "tags": []
   },
   "outputs": [],
   "source": [
    "def d2minAndDeltarto2(Ntot, Lx, neigh_dict, r_dict, xy_pairs_next):\n",
    "    '''Calculates d2min for each particle at a certain MDstep\n",
    "    Parameters:\n",
    "    Lx; float; x-length of system\n",
    "    Ntot; int; total number of particles\n",
    "    xy_pairs_next; dictionary; returned id_pos_list (by using dump file of the next MD step )\n",
    "    neigh_dict; dictionary; \n",
    "                key: particle ID (int), value: (sorted) lists of neighbor particle IDs.\n",
    "    r_dict; dictionary; \n",
    "                key: particle ID (int); value: lists of tuples (x,y) of neighbor particles.\n",
    "                                                Tuples sorted by neighbor IDs. \n",
    "    Returns:\n",
    "    d2min_falk,Deltarto2\n",
    "    d2min_falk; dictionary; key: particle ID (int), value: d2min (float)\n",
    "    '''\n",
    "    # X & Y matrices:\n",
    "    Xmat_xx = np.zeros(Ntot)\n",
    "    Xmat_xy = np.zeros(Ntot)\n",
    "    Xmat_yx = np.zeros(Ntot)\n",
    "    Xmat_yy = np.zeros(Ntot)\n",
    "    Ymat_xx = np.zeros(Ntot)\n",
    "    Ymat_xy = np.zeros(Ntot)\n",
    "    Ymat_yx = np.zeros(Ntot)\n",
    "    Ymat_yy = np.zeros(Ntot)\n",
    "    Kmatxx = np.zeros(Ntot)\n",
    "    Kmatxy = np.zeros(Ntot)\n",
    "    Kmatyx = np.zeros(Ntot)\n",
    "    Kmatyy = np.zeros(Ntot)\n",
    "    \n",
    "    \n",
    "    r_dict_next = {} # initiating r_dict_next for the dump file from the next MD step\n",
    "    for i in range(1, Ntot+1): \n",
    "        r_dict_next[i] = [] # initiating an empty list for each particle ID\n",
    "         \n",
    "    # This loop determines Xmat, Ymat, Kmat\n",
    "    for i in range(1, Ntot+1):\n",
    "        # getting position of particle i\n",
    "        xi = xy_pairs_next[i][0]\n",
    "        yi = xy_pairs_next[i][1]\n",
    "        \n",
    "        for k in range(len(neigh_dict[i])):\n",
    "            j = neigh_dict[i][k]\n",
    "            if j>i : \n",
    "                # we have already updated matrices for j<i neighbors in previous iterations\n",
    "                # go through the full codes below; you will notice \n",
    "                \n",
    "                # getting distance between particle and neighbor at reference MD step; \n",
    "                #     xj(t-Delta t)-xi(t-Delta t),yj(t-Delta t)-yi(t-Delta t)\n",
    "                xr_ref = r_dict[i][k][0]\n",
    "                yr_ref = r_dict[i][k][1]\n",
    "\n",
    "                # getting position of neighbor j at next MD step; x(t),y(t)\n",
    "                xj = xy_pairs_next[j][0]\n",
    "                yj = xy_pairs_next[j][1]\n",
    "                # getting distance between particle and neighbor in the next MD step; xj(t)-xi(t),yj(t)-yi(t)\n",
    "                xr = xj-xi\n",
    "                yr = yj-yi\n",
    "                if xr > Lx/2: xr -= Lx\n",
    "                elif xr < - Lx/2: xr += Lx\n",
    "            \n",
    "                # updating r_dict_next for both i and j\n",
    "                r_dict_next[i].append( (xr, yr) )\n",
    "                r_dict_next[j].append( (-xr, -yr) )\n",
    "            \n",
    "                # updating matrix elements of X and Y\n",
    "                # i is the particle ID, i-1 is the index number;\n",
    "                # list index starts from 0 but particle ID starts from 1\n",
    "                Xmat_xx[i-1] += xr * xr_ref \n",
    "                Xmat_xy[i-1] += xr * yr_ref \n",
    "                Xmat_yx[i-1] += yr * xr_ref\n",
    "                Xmat_yy[i-1] += yr * yr_ref\n",
    "                Ymat_xx[i-1] += xr_ref * xr_ref\n",
    "                Ymat_xy[i-1] += xr_ref * yr_ref\n",
    "                Ymat_yx[i-1] += yr_ref * xr_ref\n",
    "                Ymat_yy[i-1] += yr_ref * yr_ref\n",
    "\n",
    "                Xmat_xx[j-1] += xr * xr_ref # for j neighbor, the added product is actually (-xr)*(-xr_ref)\n",
    "                # same (-1)*(-1) for the rest\n",
    "                Xmat_xy[j-1] += xr * yr_ref \n",
    "                Xmat_yx[j-1] += yr * xr_ref\n",
    "                Xmat_yy[j-1] += yr * yr_ref\n",
    "                Ymat_xx[j-1] += xr_ref * xr_ref\n",
    "                Ymat_xy[j-1] += xr_ref * yr_ref\n",
    "                Ymat_yx[j-1] += yr_ref * xr_ref\n",
    "                Ymat_yy[j-1] += yr_ref * yr_ref\n",
    "    for i in range(1, Ntot+1):\n",
    "        # follow notes from Katharina Vollmayr-Lee and Falk and Langer Paper for d2min derivation\n",
    "        onedivdetyi = 1/(Ymat_xx[i-1]*Ymat_yy[i-1] - Ymat_xy[i-1]*Ymat_yx[i-1])\n",
    "        yinvxxi = Ymat_yy[i-1] * onedivdetyi\n",
    "        yinvxyi = -Ymat_xy[i-1] * onedivdetyi\n",
    "        yinvyxi = -Ymat_yx[i-1] * onedivdetyi\n",
    "        yinvyyi = Ymat_xx[i-1] * onedivdetyi\n",
    "        Kmatxx[i-1] = Xmat_xx[i-1] * yinvxxi + Xmat_xy[i-1] * yinvyxi\n",
    "        Kmatxy[i-1] = Xmat_xx[i-1] * yinvxyi + Xmat_xy[i-1] * yinvyyi\n",
    "        Kmatyx[i-1] = Xmat_yx[i-1] * yinvxxi + Xmat_yy[i-1] * yinvyxi\n",
    "        Kmatyy[i-1] = Xmat_yx[i-1] * yinvxyi + Xmat_yy[i-1] * yinvyyi\n",
    "    \n",
    "    d2min_falk = {} # initiating a dictionary to catch d2min against the particle ID as key\n",
    "    Deltarto2 = {}\n",
    "\n",
    "    # This loop calculates D2min: \n",
    "    for i in range(1,Ntot+1):\n",
    "        d2min_falk[i] = 0\n",
    "        Deltarto2[i] = 0\n",
    "        for k in range(len(neigh_dict[i])): # iterating over all neighbors of i\n",
    "            \n",
    "            xr_ref = r_dict[i][k][0]\n",
    "            yr_ref = r_dict[i][k][1]\n",
    "            xr = r_dict_next[i][k][0]\n",
    "            yr = r_dict_next[i][k][1]\n",
    "\n",
    "            tnuxi = xr - Kmatxx[i-1] * xr_ref - Kmatxy[i-1] * yr_ref\n",
    "            tnuyi = yr - Kmatyx[i-1] * xr_ref - Kmatyy[i-1] * yr_ref\n",
    "            d2min_falk[i] += tnuxi * tnuxi + tnuyi * tnuyi\n",
    "            \n",
    "            tnuxiK1 = xr - xr_ref\n",
    "            tnuyiK1 = yr - yr_ref\n",
    "            Deltarto2[i] += tnuxiK1 * tnuxiK1 + tnuyiK1 * tnuyiK1\n",
    "\n",
    "\n",
    "#     d2min_mean = round_to_6(sum(d2min_falk.values()) / len(d2min_falk))\n",
    "    \n",
    "    return d2min_falk,Deltarto2\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a068eea6",
   "metadata": {
    "id": "a068eea6"
   },
   "source": [
    "file = open( \"d2min_1e-6.txt\", \"rb+\") #write-binary format\n",
    "\n",
    "d2min_extra = pickle.load(file) \n",
    "\n",
    "file.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a08583c6",
   "metadata": {
    "id": "a08583c6"
   },
   "source": [
    "# Calculating d2min"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "0e8d9426-c44b-4b80-8308-ef5d04b2627d",
   "metadata": {
    "id": "0e8d9426-c44b-4b80-8308-ef5d04b2627d",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0000000000000002e-06\n",
      "27000\n",
      "2700\n",
      "[500400, 503100, 505800, 508500, 511200, 513900, 516600, 519300, 522000, 524700, 527400]\n"
     ]
    }
   ],
   "source": [
    "''' Only next few lines you might need to change \n",
    "'''\n",
    "    \n",
    "gammadotbeforepoint_list = [1]\n",
    "gammadot_list = [4] \n",
    "pin_list = [16]\n",
    "phi_list = [0.845]\n",
    "total_dump_files = 11 # Number of dump files we are analyzing #usually 101, here for testing fewer\n",
    "inf_strain_min = 0.5004\n",
    "inf_strain_max = 0.5274  #usually 0.1 here for testing because 11 instead of 101 # total amount of strain occured in these dump files\n",
    "\n",
    "# this could be any MD step; we are only using this variable for getting the information of Ntot and Lx, \n",
    "# which do not change over MD steps according to the current setting\n",
    "file_str = 'confdumpallinfMD' # could be 'confdumpallelastic' or anything you name for the dump files\n",
    "\n",
    "\n",
    "# Do not change the sequence of codes from here on\n",
    "for i in range(len(gammadotbeforepoint_list)):\n",
    "    for j in range(len(gammadot_list)):\n",
    "\n",
    "        gammadot = gammadotbeforepoint_list[i]*10**(-(gammadot_list[j])) # shear rate\n",
    "        ref_MD_step = round(inf_strain_min / (gammadot*0.01) )\n",
    "        ref_file = f'{file_str}{ref_MD_step}.data'\n",
    "        \n",
    "        d2min_dict = {}\n",
    "        d2min_dict[gammadot] = {} # initiating empty dictionary against key of gammadot\n",
    "        Deltar_dict = {}     #potential future normalization: distances as is, i.e. including affine part\n",
    "        Deltar_dict[gammadot] = {}\n",
    "        \n",
    "        strain_per_line = gammadot*0.01 # strain in one MD step\n",
    "        \n",
    "\n",
    "        print(strain_per_line) \n",
    "\n",
    "        total_MDsteps = round((inf_strain_max - inf_strain_min)/strain_per_line) # total MD steps ran in between the first and last dump files\n",
    "        print(total_MDsteps)\n",
    "\n",
    "        dump_frequency =  round(total_MDsteps/(total_dump_files-1)) # how many MD steps are these dump files apart?\n",
    "        print(dump_frequency)\n",
    "\n",
    "        MDstep_list = [int(ref_MD_step + iMDl*dump_frequency) for iMDl in range(total_dump_files)] # list of MD steps of dump files\n",
    "        print(MDstep_list)\n",
    "\n",
    "        for k in range(len(pin_list)):\n",
    "\n",
    "            pin_num = pin_list[k]\n",
    "            d2min_dict[gammadot][pin_num]={} # initiating empty dictionary against key of pin_num\n",
    "            Deltar_dict[gammadot][pin_num]={}\n",
    "\n",
    "            for l in range(len(phi_list)):\n",
    "\n",
    "                phi_val = phi_list[l]\n",
    "                path = path_name(gammadot, pin_num, phi_val)\n",
    "                # go-to-function: path_name\n",
    "                Ntot, Lx = system_info(path, ref_file)\n",
    "                # go-to-function: system_info\n",
    "                d2min_dict[gammadot][pin_num][phi_val]={} # initiating empty dictionary against key of phi_val\n",
    "                Deltar_dict[gammadot][pin_num][phi_val]={}\n",
    "\n",
    "                for m in range(len(MDstep_list)-1):\n",
    "                    \n",
    "                    file = f'{file_str}{MDstep_list[m]}.data'\n",
    "                    xy_pairs0 = id_pos_list(Ntot, MDstep_list[m], path, file)\n",
    "                    # go-to-function: id_pos_list\n",
    "                    \n",
    "                    neigh_dict, r_dict = neighbor(Lx, Ntot, xy_pairs0)\n",
    "                    # go-to-function: neighbor\n",
    "                    \n",
    "                    file = f'{file_str}{MDstep_list[m+1]}.data'\n",
    "                    xy_pairs_next = id_pos_list(Ntot, MDstep_list[m+1], path, file)\n",
    "                    # go-to-function: id_pos_list\n",
    "                    \n",
    "                    # updating d2min_dict for specific MDstep\n",
    "                    d2min_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]],\\\n",
    "                     Deltar_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]] = \\\n",
    "                                                   d2minAndDeltarto2(Ntot, Lx, neigh_dict, r_dict, xy_pairs_next)\n",
    "                    \n",
    "#                     # add following lines for D2min testing:\n",
    "#                     filed2mintest = 'd2minviapython_' + str(m+1)\n",
    "#                     fileoutd2mintest = open(filed2mintest,mode='w')\n",
    "#                     for iD2 in range(1,Ntot+1):\n",
    "#                         print(iD2,\" \",d2min_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]][iD2],file=fileoutd2mintest)\n",
    "#                     fileoutd2mintest.close()\n",
    "#                     # add following lines for Deltar^2 testing:\n",
    "#                     fileDeltarto2test = 'Deltarto2viapython_' + str(m+1)\n",
    "#                     fileoutDeltarto2test = open(fileDeltarto2test,mode='w')\n",
    "#                     for iD2 in range(1,Ntot+1):\n",
    "#                         print(iD2,\" \",Deltar_dict[gammadot][pin_num][phi_val][MDstep_list[m+1]][iD2],file=fileoutDeltarto2test)\n",
    "#                     fileoutDeltarto2test.close()\n",
    "        \n",
    "        # saving d2min_dict for specific shear rate\n",
    "        newfile = 'd2min_' + str(gammadot) + '.txt'  \n",
    "        with open(newfile, 'wb') as f:\n",
    "            pickle.dump(d2min_dict, f, pickle.HIGHEST_PROTOCOL)  \n",
    "\n",
    "        # saving Deltar_dict for specific shear rate\n",
    "        newfile2 = 'Deltar_' + str(gammadot) + '.txt'  \n",
    "        with open(newfile2, 'wb') as f2:\n",
    "            pickle.dump(Deltar_dict, f2, pickle.HIGHEST_PROTOCOL)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "132f68b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "13\n",
      "22\n",
      "26\n",
      "32\n",
      "39\n",
      "47\n",
      "56\n",
      "63\n",
      "66\n",
      "66\n"
     ]
    }
   ],
   "source": [
    "for shear_rate in d2min_dict:\n",
    "    for pin in d2min_dict[shear_rate]:\n",
    "        for phi in d2min_dict[shear_rate][pin]:\n",
    "            num_events = 0\n",
    "            for mdstep in d2min_dict[shear_rate][pin][phi]:\n",
    "                d2min_mdstep = d2min_dict[shear_rate][pin][phi][mdstep]\n",
    "                for particle in d2min_mdstep:\n",
    "                    if d2min_mdstep[particle]>1:\n",
    "                        num_events += 1\n",
    "                       \n",
    "                print(num_events)\n",
    "                    \n",
    "                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dbb3aef8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}